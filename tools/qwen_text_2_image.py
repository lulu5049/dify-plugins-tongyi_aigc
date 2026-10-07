# author: sawyer-shi

import json
import logging
from collections.abc import Generator
from typing import Any

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from tools.bailian_endpoints import native_base_url

logger = logging.getLogger(__name__)

QWEN_IMAGE_2_SERIES_MODELS = {
    "qwen-image-2.0",
    "qwen-image-2.0-2026-03-03",
    "qwen-image-2.0-pro",
    "qwen-image-2.0-pro-2026-03-03",
}

QWEN_IMAGE_LEGACY_SIZE_OPTIONS = {
    "1664*928",
    "1472*1104",
    "1328*1328",
    "1104*1472",
    "928*1664",
}


class QwenText2ImageTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        Tongyi Qwen text-to-image tool.
        """
        logger.info("Starting qwen text-to-image task")

        try:
            api_key = self.runtime.credentials.get("api_key")
            if not api_key:
                msg = "❌ API密钥未配置"
                logger.error(msg)
                yield self.create_text_message(msg)
                return

            api_url = f"{native_base_url(self.runtime.credentials)}/services/aigc/multimodal-generation/generation"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            prompt = tool_parameters.get("prompt", "").strip()
            if not prompt:
                msg = "❌ 请输入提示词"
                logger.warning(msg)
                yield self.create_text_message(msg)
                return

            if len(prompt) > 800:
                prompt = prompt[:800]

            model = tool_parameters.get("model", "qwen-image-2.0-pro")
            negative_prompt = tool_parameters.get("negative_prompt", "")
            if negative_prompt:
                negative_prompt = negative_prompt[:500]
            prompt_extend = tool_parameters.get("prompt_extend")
            watermark = tool_parameters.get("watermark")
            size = tool_parameters.get("size", "")
            seed = tool_parameters.get("seed")
            n = tool_parameters.get("n")

            if not size:
                size = "2048*2048" if model in QWEN_IMAGE_2_SERIES_MODELS else "1664*928"

            if model in QWEN_IMAGE_2_SERIES_MODELS:
                if not self._is_valid_qwen_image_2_size(size):
                    msg = "❌ qwen-image-2.0系列的size总像素需在512*512到2048*2048之间"
                    logger.warning(msg)
                    yield self.create_text_message(msg)
                    return
            elif size not in QWEN_IMAGE_LEGACY_SIZE_OPTIONS:
                msg = "❌ 当前模型仅支持固定分辨率：1664*928/1472*1104/1328*1328/1104*1472/928*1664"
                logger.warning(msg)
                yield self.create_text_message(msg)
                return

            yield self.create_text_message("🚀 文生图任务启动中...")
            yield self.create_text_message(f"🤖 使用模型: {model}")
            yield self.create_text_message(
                f"📝 提示词: {prompt[:50]}{'...' if len(prompt) > 50 else ''}"
            )
            if size:
                yield self.create_text_message(f"📐 图像尺寸: {size}")
            yield self.create_text_message("⏳ 正在连接通义API...")

            payload: dict[str, Any] = {
                "model": model,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"text": prompt},
                            ],
                        }
                    ]
                },
                "parameters": {},
            }

            if negative_prompt:
                payload["parameters"]["negative_prompt"] = negative_prompt.strip()
            if prompt_extend is not None:
                payload["parameters"]["prompt_extend"] = prompt_extend
            if watermark is not None:
                payload["parameters"]["watermark"] = watermark
            payload["parameters"]["size"] = size
            if n is not None:
                try:
                    n_value = int(n)
                except (TypeError, ValueError):
                    n_value = 1

                if model in QWEN_IMAGE_2_SERIES_MODELS:
                    if n_value < 1:
                        n_value = 1
                    if n_value > 6:
                        n_value = 6
                else:
                    n_value = 1
                payload["parameters"]["n"] = n_value
            if seed is not None:
                try:
                    payload["parameters"]["seed"] = int(seed)
                except (TypeError, ValueError):
                    pass

            logger.info("Submitting request: %s", json.dumps(payload, ensure_ascii=False))
            yield self.create_text_message("🎨 正在生成图像，请稍候...")

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
                logger.error("API status %s: %s", response.status_code, response.text[:300])
                yield self.create_text_message(f"❌ API 响应状态码: {response.status_code}")
                if response.text:
                    yield self.create_text_message(f"🔧 响应内容: {response.text[:500]}")
                return

            try:
                resp_data = response.json()
            except json.JSONDecodeError as e:
                logger.error("Failed to parse JSON: %s - %s", str(e), response.text[:300])
                yield self.create_text_message("❌ API 响应解析失败（非JSON）")
                return

            output = resp_data.get("output", {})
            choices = output.get("choices", [])
            if not choices:
                yield self.create_text_message("❌ API 响应中未返回图像数据")
                return

            yield self.create_text_message("🎉 图像生成成功！")

            for i, choice in enumerate(choices):
                message = choice.get("message", {})
                content = message.get("content", [])
                image_url = ""
                for item in content:
                    if isinstance(item, dict) and "image" in item:
                        image_url = item.get("image", "")
                        break
                if not image_url:
                    yield self.create_text_message(f"❌ 未获取到第 {i + 1} 张图片的URL")
                    return
                yield self.create_image_message(image_url)
                yield self.create_text_message(f"✅ 第 {i + 1} 张图片生成完成！")

            usage = resp_data.get("usage", {})
            if usage:
                if isinstance(usage, dict):
                    yield self.create_text_message("📊 使用统计:")
                    for key, value in usage.items():
                        yield self.create_text_message(f"  - {key}: {value}")
                else:
                    try:
                        usage_text = json.dumps(usage, ensure_ascii=False)
                    except Exception:
                        usage_text = str(usage)
                    yield self.create_text_message(f"📊 使用信息: {usage_text}")

            yield self.create_text_message("🎯 文生图任务完成！")
            logger.info("Qwen text-to-image task completed")

        except Exception as e:
            error_msg = f"❌ 生成图像时出现未预期错误: {str(e)}"
            logger.exception(error_msg)
            yield self.create_text_message(error_msg)

    @staticmethod
    def _is_valid_qwen_image_2_size(size: str) -> bool:
        if not size or "*" not in size:
            return False
        try:
            width_text, height_text = size.split("*", 1)
            width = int(width_text)
            height = int(height_text)
        except (TypeError, ValueError):
            return False

        if width <= 0 or height <= 0:
            return False

        pixels = width * height
        return 512 * 512 <= pixels <= 2048 * 2048
