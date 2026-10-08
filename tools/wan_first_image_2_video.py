# author: sawyer-shi

import base64
import json
import logging
from collections.abc import Generator
from io import BytesIO
from typing import Any

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from PIL import Image

from tools.bailian_endpoints import native_base_url
from tools.bailian_task_waiter import maybe_wait_for_video
from tools.video_model_compat import is_wan3, normalize_i2v_parameters

logger = logging.getLogger(__name__)


class WanFirstImage2VideoTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        Tongyi Wanxiang image-to-video tool (async submit).
        """
        logger.info("Starting wan image-to-video task")

        try:
            model = tool_parameters.get("model", "wan2.6-i2v").strip()
            is_wan30 = is_wan3(model)
            is_wan27_i2v = model.startswith("wan2.7-i2v")
            uses_media_protocol = is_wan30 or is_wan27_i2v
            api_key = self.runtime.credentials.get("api_key")
            if not api_key:
                msg = "❌ API密钥未配置"
                logger.error(msg)
                yield self.create_text_message(msg)
                return

            image_obj = tool_parameters.get("image_input")
            img_url_str = tool_parameters.get("img_url")
            processed_img = self._process_image(image_obj if image_obj else img_url_str)
            audio_url = tool_parameters.get("audio_url", "").strip()

            payload: dict[str, Any]
            if uses_media_protocol:
                last_frame_obj = tool_parameters.get("last_frame_input")
                last_frame_url = tool_parameters.get("last_frame_url")
                processed_last_frame = self._process_image(
                    last_frame_obj if last_frame_obj else last_frame_url
                )
                first_clip_url = tool_parameters.get("first_clip_url", "").strip()

                media: list[dict[str, str]] = []
                if is_wan30 and first_clip_url:
                    yield self.create_text_message(
                        "❌ Wan 3.0 不支持 first_clip 视频续写，请改用 wan2.7-i2v 的视频续写工具。"
                    )
                    return
                if first_clip_url and is_wan27_i2v:
                    media.append({"type": "first_clip", "url": first_clip_url})
                    if processed_last_frame:
                        media.append({"type": "last_frame", "url": processed_last_frame})
                    if audio_url:
                        yield self.create_text_message(
                            "ℹ️ wan2.7-i2v 的视频续写模式不支持 driving_audio，已忽略 audio_url。"
                        )
                else:
                    if not processed_img:
                        yield self.create_text_message("❌ 请上传有效的首帧图片")
                        return
                    media.append({"type": "first_frame", "url": processed_img})
                    if processed_last_frame:
                        media.append({"type": "last_frame", "url": processed_last_frame})
                    if audio_url and is_wan27_i2v:
                        media.append({"type": "driving_audio", "url": audio_url})

                payload = {
                    "model": model,
                    "input": {
                        "media": media,
                    },
                    "parameters": {},
                }
            else:
                if not processed_img:
                    yield self.create_text_message("❌ 请上传有效的图片")
                    return
                payload = {
                    "model": model,
                    "input": {
                        "img_url": processed_img,
                    },
                    "parameters": {},
                }

            input_params = payload["input"]
            prompt = tool_parameters.get("prompt", "").strip()
            if prompt:
                if is_wan30:
                    limit = 20000
                elif is_wan27_i2v:
                    limit = 5000
                else:
                    limit = 1500 if "wan2.6" in model or "wan2.5" in model else 800
                input_params["prompt"] = prompt[:limit]

            negative_prompt = tool_parameters.get("negative_prompt", "").strip()
            if negative_prompt and not is_wan30:
                input_params["negative_prompt"] = negative_prompt[:500]

            if audio_url and ("wan2.6" in model or "wan2.5" in model):
                input_params["audio_url"] = audio_url

            params, compatibility_notes = normalize_i2v_parameters(model, tool_parameters)
            payload["parameters"] = params
            for note in compatibility_notes:
                yield self.create_text_message(f"ℹ️ {note}")

            template = tool_parameters.get("template", "").strip()
            if template and not uses_media_protocol:
                params["template"] = template
            elif template and uses_media_protocol:
                yield self.create_text_message(f"ℹ️ {model} 不使用 template 参数，已忽略。")

            api_url = f"{native_base_url(self.runtime.credentials)}/services/aigc/video-generation/video-synthesis"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable",
            }

            debug_payload = json.loads(json.dumps(payload))
            if "img_url" in debug_payload.get("input", {}):
                if len(debug_payload["input"]["img_url"]) > 200:
                    debug_payload["input"]["img_url"] = "data:image/...[Base64 Hidden]"
            for media_item in debug_payload.get("input", {}).get("media", []):
                media_url = media_item.get("url", "")
                if isinstance(media_url, str) and len(media_url) > 200:
                    media_type = media_item.get("type", "media")
                    if media_type in {"first_frame", "last_frame"}:
                        media_item["url"] = "data:image/...[Base64 Hidden]"
                    elif media_type == "driving_audio":
                        media_item["url"] = "data:audio/...[Hidden]"
                    elif media_type == "first_clip":
                        media_item["url"] = "https://.../[Video URL Hidden]"
                    else:
                        media_item["url"] = "...[Hidden]"
            logger.info("Request Payload: %s", json.dumps(debug_payload, ensure_ascii=False))

            yield self.create_text_message("🚀 图生视频任务启动中...")
            yield self.create_text_message(f"🤖 使用模型: {model}")
            yield self.create_text_message("⏳ 正在连接通义API...")

            try:
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
            except requests.exceptions.Timeout:
                msg = "❌ 请求超时，请稍后重试"
                logger.error(msg)
                yield self.create_text_message(msg)
                return
            except requests.exceptions.RequestException as e:
                msg = f"❌ 请求失败: {str(e)}"
                logger.error(msg)
                yield self.create_text_message(msg)
                return

            if response.status_code != 200:
                yield self.create_text_message(f"❌ API 响应状态码: {response.status_code}")
                if response.text:
                    yield self.create_text_message(f"🔧 响应内容: {response.text[:500]}")
                return

            try:
                result_data = response.json()
            except json.JSONDecodeError:
                yield self.create_text_message("❌ API 响应解析失败（非JSON）")
                return

            handled = yield from maybe_wait_for_video(
                self,
                self.runtime.credentials,
                result_data,
                tool_parameters,
            )
            if handled:
                return

            if "output" in result_data:
                yield self.create_text_message(self._format_success_message(result_data))
                yield self.create_json_message(result_data)
            else:
                yield self.create_text_message(
                    f"❌ 响应格式异常: {response.text[:500]}"
                )

            logger.info("Wan image-to-video task submitted")

        except Exception as e:
            error_msg = f"❌ 生成视频时出现未预期错误: {str(e)}"
            logger.exception(error_msg)
            yield self.create_text_message(error_msg)

    @staticmethod
    def _format_success_message(result_data: dict[str, Any]) -> str:
        task_id = result_data.get("output", {}).get("task_id", "unknown")
        task_status = result_data.get("output", {}).get("task_status", "unknown")
        return (
            f"🎬 视频任务已提交\nID: {task_id}\nStatus: {task_status}\n"
            "请稍后查询结果。"
        )

    @staticmethod
    def _process_image(image_data: Any) -> str:
        if not image_data:
            return ""

        if isinstance(image_data, str) and (
            image_data.startswith("http") or image_data.startswith("data:")
        ):
            return image_data.strip()

        image_bytes = None
        if hasattr(image_data, "blob"):
            image_bytes = image_data.blob
        elif hasattr(image_data, "read"):
            image_bytes = image_data.read()
        elif isinstance(image_data, bytes):
            image_bytes = image_data

        if not image_bytes:
            return ""

        try:
            if len(image_bytes) > 10 * 1024 * 1024:
                return ""

            img = Image.open(BytesIO(image_bytes))

            output_fmt = img.format
            if img.mode in ("RGBA", "LA") or (
                img.mode == "P" and "transparency" in img.info
            ):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                bg.paste(img, mask=img.split()[3])
                img = bg
                output_fmt = "JPEG"
            elif img.mode != "RGB":
                img = img.convert("RGB")
                if not output_fmt:
                    output_fmt = "JPEG"

            buffer = BytesIO()
            save_fmt = output_fmt if output_fmt in ["JPEG", "PNG", "WEBP"] else "JPEG"
            img.save(buffer, format=save_fmt, quality=95)

            b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            mime = "image/jpeg" if save_fmt == "JPEG" else f"image/{save_fmt.lower()}"
            return f"data:{mime};base64,{b64_str}"
        except Exception as e:
            logger.error("Image processing failed: %s", str(e))
            return ""
