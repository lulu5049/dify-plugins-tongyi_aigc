# author: sawyer-shi

import base64
import json
import logging
from collections.abc import Generator
from typing import Any

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from tools.bailian_endpoints import native_base_url
from tools.bailian_task_waiter import maybe_wait_for_video
from tools.video_model_compat import is_wan3, normalize_reference_parameters

logger = logging.getLogger(__name__)


class WanReferenceVideoTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        Tongyi Wanxiang reference-to-video tool (async submit).
        """
        logger.info("Starting wan reference-to-video task")

        try:
            model = tool_parameters.get("model", "wan2.6-r2v").strip()
            is_wan30 = is_wan3(model)
            is_wan27 = model.startswith("wan2.7-")
            uses_media_protocol = is_wan30 or is_wan27
            api_key = self.runtime.credentials.get("api_key")
            if not api_key:
                msg = "❌ API密钥未配置"
                logger.error(msg)
                yield self.create_text_message(msg)
                return

            reference_urls_str = tool_parameters.get("reference_urls", "").strip()
            if not reference_urls_str:
                yield self.create_text_message("❌ 请提供参考文件URL")
                return

            reference_urls = [
                url.strip()
                for url in reference_urls_str.split(";")
                if url.strip()
            ]
            if len(reference_urls) > (15 if is_wan30 else 5):
                yield self.create_text_message(
                    "❌ Wan 3.0 最多支持10张参考图 + 5段参考视频；Wan 2.7/2.6 此工具最多支持5个参考素材。"
                )
                return

            processed_urls: list[str] = []
            for ref_url in reference_urls:
                processed_url = self._process_media(ref_url)
                if not processed_url:
                    yield self.create_text_message(f"❌ 无效的参考URL: {ref_url}")
                    return
                processed_urls.append(processed_url)

            prompt = tool_parameters.get("prompt", "").strip()
            if not prompt:
                yield self.create_text_message("❌ 请输入提示词")
                return

            prompt_limit = 20000 if is_wan30 else (5000 if is_wan27 else 1500)
            payload: dict[str, Any] = {
                "model": model,
                "input": {
                    "prompt": prompt[:prompt_limit],
                },
                "parameters": {},
            }

            input_params = payload["input"]
            if uses_media_protocol:
                media: list[dict[str, str]] = []
                for media_url in processed_urls:
                    media_type = self._infer_media_type(media_url)
                    media.append({"type": media_type, "url": media_url})

                first_frame_image = self._process_media(
                    tool_parameters.get("first_frame_image", "")
                )
                if first_frame_image and is_wan30:
                    yield self.create_text_message(
                        "ℹ️ Wan 3.0 的参考生视频模式不能与 first_frame 混用，已忽略首帧参数。"
                    )
                elif first_frame_image:
                    media.append({"type": "first_frame", "url": first_frame_image})

                if len([m for m in media if m["type"] == "first_frame"]) > 1:
                    yield self.create_text_message("❌ first_frame 最多支持1张")
                    return

                visual_reference_count = len(
                    [
                        m
                        for m in media
                        if m["type"] in {"reference_image", "reference_video"}
                    ]
                )
                if visual_reference_count == 0:
                    yield self.create_text_message(f"❌ {model} 至少需要1个参考图像或视频")
                    return
                if is_wan27 and visual_reference_count > 5:
                    yield self.create_text_message("❌ wan2.7-r2v 的图像+视频总数不能超过5")
                    return
                if is_wan30:
                    video_count = len([m for m in media if m["type"] == "reference_video"])
                    image_count = len([m for m in media if m["type"] == "reference_image"])
                    if video_count > 5 or image_count > 10:
                        yield self.create_text_message("❌ Wan 3.0 最多支持5段参考视频、10张参考图片")
                        return

                input_params["media"] = media

                reference_voice = self._process_audio(tool_parameters.get("reference_voice"))
                if reference_voice and is_wan30:
                    media.append({"type": "reference_audio", "url": reference_voice})
                elif reference_voice:
                    input_params["reference_voice"] = reference_voice
            else:
                input_params["reference_urls"] = processed_urls

            negative_prompt = tool_parameters.get("negative_prompt", "").strip()
            if negative_prompt and not is_wan30:
                input_params["negative_prompt"] = negative_prompt[:500]

            has_reference_video = any(
                self._infer_media_type(media_url) == "reference_video"
                for media_url in processed_urls
            )
            params, compatibility_notes = normalize_reference_parameters(
                model, tool_parameters, has_reference_video
            )
            payload["parameters"] = params
            for note in compatibility_notes:
                yield self.create_text_message(f"ℹ️ {note}")

            api_url = f"{native_base_url(self.runtime.credentials)}/services/aigc/video-generation/video-synthesis"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable",
            }

            debug_payload = json.loads(json.dumps(payload))
            if "reference_urls" in debug_payload.get("input", {}):
                for i, url in enumerate(debug_payload["input"]["reference_urls"]):
                    if len(url) > 200:
                        debug_payload["input"]["reference_urls"][
                            i
                        ] = "data:video/...[Base64 Hidden]"
            if "media" in debug_payload.get("input", {}):
                for item in debug_payload["input"]["media"]:
                    if len(item.get("url", "")) > 200:
                        item["url"] = "data:media/...[Base64 Hidden]"
            if "reference_voice" in debug_payload.get("input", {}) and len(
                debug_payload["input"]["reference_voice"]
            ) > 200:
                debug_payload["input"]["reference_voice"] = (
                    "data:audio/...[Base64 Hidden]"
                )
            logger.info("Request Payload: %s", json.dumps(debug_payload, ensure_ascii=False))

            yield self.create_text_message("🚀 参考生视频任务启动中...")
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

            logger.info("Wan reference-to-video task submitted")

        except Exception as e:
            error_msg = f"❌ 生成视频时出现未预期错误: {str(e)}"
            logger.exception(error_msg)
            yield self.create_text_message(error_msg)

    @staticmethod
    def _format_success_message(result_data: dict[str, Any]) -> str:
        task_id = result_data.get("output", {}).get("task_id", "unknown")
        task_status = result_data.get("output", {}).get("task_status", "unknown")
        return (
            f"🎬 参考生视频任务已提交\nID: {task_id}\nStatus: {task_status}\n"
            "请稍后查询结果。"
        )

    @staticmethod
    def _map_size_to_wan27(size: Any) -> tuple[str, str]:
        normalized_size = str(size).strip().lower().replace(" ", "")
        mapping = {
            "1280*720": ("720P", "16:9"),
            "720*1280": ("720P", "9:16"),
            "960*960": ("720P", "1:1"),
            "1104*832": ("720P", "4:3"),
            "832*1104": ("720P", "3:4"),
            "1920*1080": ("1080P", "16:9"),
            "1080*1920": ("1080P", "9:16"),
            "1440*1440": ("1080P", "1:1"),
            "1648*1248": ("1080P", "4:3"),
            "1248*1648": ("1080P", "3:4"),
            "1088*832": ("720P", "4:3"),
            "832*1088": ("720P", "3:4"),
            "1632*1248": ("1080P", "4:3"),
            "1248*1632": ("1080P", "3:4"),
        }
        return mapping.get(normalized_size, ("", ""))

    @staticmethod
    def _infer_media_type(media_url: str) -> str:
        media_url_lower = str(media_url).lower()
        if media_url_lower.startswith("data:video/"):
            return "reference_video"
        if media_url_lower.startswith("data:image/"):
            return "reference_image"
        if any(
            media_url_lower.split("?")[0].endswith(ext)
            for ext in (".mp4", ".mov", ".m4v", ".webm")
        ):
            return "reference_video"
        return "reference_image"

    @staticmethod
    def _process_media(media_data: Any) -> str:
        if not media_data:
            return ""

        if isinstance(media_data, str) and (
            media_data.startswith("http")
            or media_data.startswith("data:")
            or media_data.startswith("oss://")
        ):
            return media_data.strip()

        media_bytes = None
        media_name = ""
        if not isinstance(media_data, str) and hasattr(media_data, "blob"):
            media_bytes = media_data.blob
            media_name = str(getattr(media_data, "filename", "")).lower()
        elif not isinstance(media_data, str) and hasattr(media_data, "read"):
            media_bytes = media_data.read()
            media_name = str(getattr(media_data, "name", "")).lower()
        elif isinstance(media_data, bytes):
            media_bytes = media_data
        elif isinstance(media_data, str):
            try:
                media_bytes = base64.b64decode(media_data)
            except Exception:
                return ""

        if not media_bytes:
            return ""

        if len(media_bytes) > 100 * 1024 * 1024:
            logger.error("Media size exceeds 100MB limit")
            return ""

        try:
            is_video = media_name.endswith((".mp4", ".mov", ".m4v", ".webm"))
            if not media_name and media_bytes[:8].startswith((b"\x00\x00\x00", b"ftyp")):
                is_video = True
            mime = "video/mp4" if is_video else "image/png"
            b64_str = base64.b64encode(media_bytes).decode("utf-8")
            return f"data:{mime};base64,{b64_str}"
        except Exception as e:
            logger.error("Media processing failed: %s", str(e))
            return ""

    @staticmethod
    def _process_audio(audio_data: Any) -> str:
        if not audio_data:
            return ""

        if isinstance(audio_data, str) and (
            audio_data.startswith("http")
            or audio_data.startswith("data:")
            or audio_data.startswith("oss://")
        ):
            return audio_data.strip()

        audio_bytes = None
        if not isinstance(audio_data, str) and hasattr(audio_data, "blob"):
            audio_bytes = audio_data.blob
        elif not isinstance(audio_data, str) and hasattr(audio_data, "read"):
            audio_bytes = audio_data.read()
        elif isinstance(audio_data, bytes):
            audio_bytes = audio_data
        elif isinstance(audio_data, str):
            try:
                audio_bytes = base64.b64decode(audio_data)
            except Exception:
                return ""

        if not audio_bytes:
            return ""
        if len(audio_bytes) > 15 * 1024 * 1024:
            logger.error("Audio size exceeds 15MB limit")
            return ""

        audio_format = "wav" if audio_bytes.startswith(b"RIFF") else "mp3"
        b64_str = base64.b64encode(audio_bytes).decode("utf-8")
        return f"data:audio/{audio_format};base64,{b64_str}"
