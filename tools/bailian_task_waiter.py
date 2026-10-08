import time
from collections.abc import Generator
from typing import Any

import requests
from dify_plugin.entities.tool import ToolInvokeMessage

from tools.bailian_endpoints import native_base_url


TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "CANCELED", "CANCELLED", "UNKNOWN"}


def wait_for_bailian_task(
    tool: Any,
    credentials: dict[str, Any],
    task_id: str,
    *,
    poll_interval_seconds: int = 10,
    max_wait_seconds: int = 540,
    heartbeat_seconds: int = 30,
) -> Generator[ToolInvokeMessage, None, dict[str, Any] | None]:
    """Poll a Bailian async task until it reaches a terminal state.

    The generator emits periodic progress messages so Dify's streaming
    connection remains active while the tool invocation is running.
    """
    api_key = credentials.get("api_key")
    if not api_key:
        yield tool.create_text_message("❌ API密钥未配置")
        return None

    poll_interval_seconds = max(3, min(int(poll_interval_seconds), 60))
    max_wait_seconds = max(30, int(max_wait_seconds))
    heartbeat_seconds = max(poll_interval_seconds, int(heartbeat_seconds))

    api_url = f"{native_base_url(credentials)}/tasks/{task_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    started = time.monotonic()
    last_status = ""
    last_heartbeat = -heartbeat_seconds

    while True:
        elapsed = int(time.monotonic() - started)
        if elapsed >= max_wait_seconds:
            yield tool.create_text_message(
                f"⌛ 已等待 {elapsed}s，任务仍未结束。Task ID: {task_id}。"
                "任务可能仍在百炼后台排队，可稍后用对应的结果查询工具继续查询。"
            )
            return None

        try:
            response = requests.get(api_url, headers=headers, timeout=30)
        except requests.exceptions.Timeout:
            if elapsed - last_heartbeat >= heartbeat_seconds:
                yield tool.create_text_message(
                    f"⏳ 查询暂时超时，任务仍在等待中（{elapsed}s）。Task ID: {task_id}"
                )
                last_heartbeat = elapsed
            time.sleep(poll_interval_seconds)
            continue
        except requests.exceptions.RequestException as exc:
            yield tool.create_text_message(f"❌ 查询任务状态失败: {exc}")
            return None

        if response.status_code != 200:
            yield tool.create_text_message(
                f"❌ 查询任务状态失败，HTTP {response.status_code}: {response.text[:500]}"
            )
            return None

        try:
            data = response.json()
        except ValueError:
            yield tool.create_text_message(
                f"❌ 查询任务状态返回非JSON内容: {response.text[:500]}"
            )
            return None

        output = data.get("output", {}) or {}
        status = str(output.get("task_status") or "UNKNOWN").upper()

        if status != last_status:
            yield tool.create_text_message(
                f"🔄 任务状态: {status}（已等待 {elapsed}s）"
            )
            last_status = status
            last_heartbeat = elapsed
        elif elapsed - last_heartbeat >= heartbeat_seconds:
            yield tool.create_text_message(
                f"⏳ 任务仍为 {status}（已等待 {elapsed}s）"
            )
            last_heartbeat = elapsed

        if status in TERMINAL_STATUSES:
            return data

        time.sleep(poll_interval_seconds)


def emit_final_video_result(
    tool: Any,
    result_data: dict[str, Any],
) -> Generator[ToolInvokeMessage, None, None]:
    output = result_data.get("output", {}) or {}
    status = str(output.get("task_status") or "UNKNOWN").upper()
    task_id = output.get("task_id", "unknown")

    if status == "SUCCEEDED":
        video_url = output.get("video_url") or ""
        last_frame_url = output.get("last_frame_url") or ""
        yield tool.create_text_message(f"✅ 视频生成完成！Task ID: {task_id}")
        if video_url:
            yield tool.create_text_message(f"🎬 视频链接: {video_url}")
        if last_frame_url:
            yield tool.create_text_message(f"🖼️ 尾帧链接: {last_frame_url}")
        yield tool.create_json_message(result_data)
        return

    error_code = output.get("code", "Unknown")
    error_message = output.get("message", "Unknown error")
    yield tool.create_text_message(
        f"❌ 视频任务失败\n"
        f"Task ID: {task_id}\n"
        f"Status: {status}\n"
        f"Error Code: {error_code}\n"
        f"Error Message: {error_message}"
    )
    yield tool.create_json_message(result_data)


def _parse_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    return default


def maybe_wait_for_video(
    tool: Any,
    credentials: dict[str, Any],
    submit_response: dict[str, Any],
    tool_parameters: dict[str, Any],
) -> Generator[ToolInvokeMessage, None, bool]:
    """Wait for a submitted video task until Bailian returns a terminal state.

    Waiting is an internal plugin behavior, not a user-facing form parameter.
    """


    output = submit_response.get("output", {}) or {}
    task_id = str(output.get("task_id") or "").strip()
    if not task_id:
        return False

    max_wait = 540

    yield tool.create_text_message(
        f"✅ 任务已提交，Task ID: {task_id}。现在由插件自动等待最终结果..."
    )
    final_data = yield from wait_for_bailian_task(
        tool,
        credentials,
        task_id,
        poll_interval_seconds=10,
        max_wait_seconds=max_wait,
        heartbeat_seconds=30,
    )
    if final_data is not None:
        yield from emit_final_video_result(tool, final_data)
    return True


def emit_final_image_result(
    tool: Any,
    result_data: dict[str, Any],
) -> Generator[ToolInvokeMessage, None, None]:
    output = result_data.get("output", {}) or {}
    status = str(output.get("task_status") or "UNKNOWN").upper()
    task_id = output.get("task_id", "unknown")

    if status == "SUCCEEDED":
        results = output.get("results", []) or []
        yield tool.create_text_message(f"✅ 图片生成完成！Task ID: {task_id}")
        for result in results:
            if isinstance(result, dict):
                image_url = result.get("url") or result.get("image_url") or ""
                if image_url:
                    yield tool.create_image_message(image_url)
        yield tool.create_json_message(result_data)
        return

    error_code = output.get("code", "Unknown")
    error_message = output.get("message", "Unknown error")
    yield tool.create_text_message(
        f"❌ 图片任务失败\n"
        f"Task ID: {task_id}\n"
        f"Status: {status}\n"
        f"Error Code: {error_code}\n"
        f"Error Message: {error_message}"
    )
    yield tool.create_json_message(result_data)


def wait_and_emit_image(
    tool: Any,
    credentials: dict[str, Any],
    submit_response: dict[str, Any],
) -> Generator[ToolInvokeMessage, None, bool]:
    output = submit_response.get("output", {}) or {}
    task_id = str(output.get("task_id") or "").strip()
    if not task_id:
        return False

    yield tool.create_text_message(
        f"✅ 任务已提交，Task ID: {task_id}。插件正在自动等待图片生成完成..."
    )
    final_data = yield from wait_for_bailian_task(
        tool,
        credentials,
        task_id,
        poll_interval_seconds=5,
        max_wait_seconds=540,
        heartbeat_seconds=30,
    )
    if final_data is not None:
        yield from emit_final_image_result(tool, final_data)
    return True
