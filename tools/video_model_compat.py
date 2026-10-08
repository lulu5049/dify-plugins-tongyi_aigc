from typing import Any


SIZE_PRESETS = {
    "480P": {
        "16:9": "832*480",
        "9:16": "480*832",
        "1:1": "624*624",
    },
    "720P": {
        "16:9": "1280*720",
        "9:16": "720*1280",
        "1:1": "960*960",
        "4:3": "1088*832",
        "3:4": "832*1088",
    },
    "1080P": {
        "16:9": "1920*1080",
        "9:16": "1080*1920",
        "1:1": "1440*1440",
        "4:3": "1632*1248",
        "3:4": "1248*1632",
    },
}

T2V_LEGACY_TIERS = {
    "wan2.6-t2v": ("720P", "1080P"),
    "wan2.5-t2v-preview": ("480P", "720P", "1080P"),
    "wan2.2-t2v-plus": ("480P", "1080P"),
    "wanx2.1-t2v-turbo": ("480P", "720P"),
    "wanx2.1-t2v-plus": ("720P",),
}

T2V_DEFAULT_SIZE = {
    "wan2.6-t2v": "1920*1080",
    "wan2.5-t2v-preview": "1920*1080",
    "wan2.2-t2v-plus": "1920*1080",
    "wanx2.1-t2v-turbo": "1280*720",
    "wanx2.1-t2v-plus": "1280*720",
}

I2V_RESOLUTIONS = {
    "wan2.6-i2v-flash": ("720P", "1080P"),
    "wan2.6-i2v": ("720P", "1080P"),
    "wan2.5-i2v-preview": ("480P", "720P", "1080P"),
    "wan2.2-i2v-flash": ("480P", "720P", "1080P"),
    "wan2.2-i2v-plus": ("480P", "1080P"),
    "wanx2.1-i2v-turbo": ("480P", "720P"),
    "wanx2.1-i2v-plus": ("720P",),
}

I2V_DEFAULT_RESOLUTION = {
    "wan2.6-i2v-flash": "1080P",
    "wan2.6-i2v": "1080P",
    "wan2.5-i2v-preview": "1080P",
    "wan2.2-i2v-flash": "720P",
    "wan2.2-i2v-plus": "1080P",
    "wanx2.1-i2v-turbo": "720P",
    "wanx2.1-i2v-plus": "720P",
}

WAN3_RATIOS = ("adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
WAN27_RATIOS = ("16:9", "9:16", "1:1", "4:3", "3:4")
HAPPYHORSE_RATIOS = ("16:9", "9:16", "1:1", "4:3", "3:4", "4:5", "5:4", "9:21", "21:9")


def is_wan3(model: str) -> bool:
    return model in {"wan3.0-video", "wan3.0-video-prime"}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _valid_seed(value: Any, allow_minus_one: bool = False) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    seed = _as_int(value, -999999)
    if allow_minus_one and seed == -1:
        return seed
    if 0 <= seed <= 2147483647:
        return seed
    return None


def _allowed_sizes(tiers: tuple[str, ...]) -> set[str]:
    return {
        size
        for tier in tiers
        for size in SIZE_PRESETS[tier].values()
    }


def normalize_t2v_parameters(model: str, p: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    params: dict[str, Any] = {}
    notes: list[str] = []

    if is_wan3(model):
        resolution = str(p.get("resolution") or "1080P").upper()
        if resolution not in {"480P", "720P", "1080P"}:
            notes.append(f"{model} 不支持 {resolution}，已改为 1080P")
            resolution = "1080P"
        ratio = str(p.get("ratio") or "adaptive")
        if ratio not in WAN3_RATIOS:
            notes.append(f"{model} 不支持宽高比 {ratio}，已改为 adaptive")
            ratio = "adaptive"
        duration = _as_int(p.get("duration"), 5)
        if duration != -1:
            duration = min(30, max(2, duration))
        params.update(resolution=resolution, ratio=ratio, duration=duration)
        if p.get("audio") is not None:
            params["audio"] = bool(p.get("audio"))
        seed = _valid_seed(p.get("seed"), allow_minus_one=True)
        if seed is not None:
            params["seed"] = seed

    elif model.startswith("wan2.7-t2v"):
        resolution = str(p.get("resolution") or "1080P").upper()
        if resolution not in {"720P", "1080P"}:
            notes.append(f"{model} 仅支持 720P/1080P，已改为 1080P")
            resolution = "1080P"
        ratio = str(p.get("ratio") or "16:9")
        if ratio not in WAN27_RATIOS:
            notes.append(f"{model} 不支持宽高比 {ratio}，已改为 16:9")
            ratio = "16:9"
        duration = min(15, max(2, _as_int(p.get("duration"), 5)))
        params.update(resolution=resolution, ratio=ratio, duration=duration)
        seed = _valid_seed(p.get("seed"))
        if seed is not None:
            params["seed"] = seed

    else:
        tiers = T2V_LEGACY_TIERS.get(model)
        if tiers:
            size = str(p.get("size") or "").replace(" ", "")
            allowed = _allowed_sizes(tiers)
            if size not in allowed:
                fixed = T2V_DEFAULT_SIZE[model]
                notes.append(f"{model} 不支持分辨率 {size or '空'}，已自动改为 {fixed}")
                size = fixed
            params["size"] = size

        if model == "wan2.6-t2v":
            params["duration"] = min(15, max(2, _as_int(p.get("duration"), 5)))
        elif model == "wan2.5-t2v-preview":
            d = _as_int(p.get("duration"), 5)
            params["duration"] = d if d in {5, 10} else 5
            if d not in {5, 10}:
                notes.append(f"{model} 时长仅支持 5/10 秒，已改为 5 秒")
        elif model in {"wan2.2-t2v-plus", "wanx2.1-t2v-turbo", "wanx2.1-t2v-plus"}:
            params["duration"] = 5
            if str(p.get("duration", "5")) != "5":
                notes.append(f"{model} 固定生成 5 秒视频")

        if model == "wan2.6-t2v" and p.get("shot_type") in {"single", "multi"}:
            params["shot_type"] = p.get("shot_type")
        if model in {"wan2.6-t2v", "wan2.5-t2v-preview"} and p.get("audio") is not None:
            params["audio"] = bool(p.get("audio"))
        seed = _valid_seed(p.get("seed"))
        if seed is not None:
            params["seed"] = seed

    if p.get("prompt_extend") is not None:
        params["prompt_extend"] = bool(p.get("prompt_extend"))
    if p.get("watermark") is not None:
        params["watermark"] = bool(p.get("watermark"))
    return params, notes


def normalize_i2v_parameters(model: str, p: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    params: dict[str, Any] = {}
    notes: list[str] = []

    if is_wan3(model):
        resolution = str(p.get("resolution") or "1080P").upper()
        if resolution not in {"480P", "720P", "1080P"}:
            resolution = "1080P"
            notes.append(f"{model} 分辨率已改为 1080P")
        ratio = str(p.get("ratio") or "adaptive")
        if ratio not in WAN3_RATIOS:
            ratio = "adaptive"
            notes.append(f"{model} 宽高比已改为 adaptive")
        duration = _as_int(p.get("duration"), 5)
        if duration != -1:
            duration = min(30, max(2, duration))
        params.update(resolution=resolution, ratio=ratio, duration=duration)
        if p.get("audio") is not None:
            params["audio"] = bool(p.get("audio"))
        seed = _valid_seed(p.get("seed"), allow_minus_one=True)
        if seed is not None:
            params["seed"] = seed

    elif model.startswith("wan2.7-i2v"):
        resolution = str(p.get("resolution") or "1080P").upper()
        if resolution not in {"720P", "1080P"}:
            resolution = "1080P"
            notes.append(f"{model} 仅支持 720P/1080P，已改为 1080P")
        params["resolution"] = resolution
        params["duration"] = min(15, max(2, _as_int(p.get("duration"), 5)))
        seed = _valid_seed(p.get("seed"))
        if seed is not None:
            params["seed"] = seed

    else:
        allowed = I2V_RESOLUTIONS.get(model)
        if allowed:
            resolution = str(p.get("resolution") or "").upper()
            if resolution not in allowed:
                fixed = I2V_DEFAULT_RESOLUTION[model]
                notes.append(f"{model} 不支持 {resolution or '空'}，已自动改为 {fixed}")
                resolution = fixed
            params["resolution"] = resolution

        if model in {"wan2.6-i2v-flash", "wan2.6-i2v"}:
            params["duration"] = min(15, max(2, _as_int(p.get("duration"), 5)))
        elif model == "wan2.5-i2v-preview":
            d = _as_int(p.get("duration"), 5)
            params["duration"] = d if d in {5, 10} else 5
            if d not in {5, 10}:
                notes.append(f"{model} 时长仅支持 5/10 秒，已改为 5 秒")
        elif model in {"wan2.2-i2v-plus", "wan2.2-i2v-flash", "wanx2.1-i2v-plus"}:
            params["duration"] = 5
        elif model == "wanx2.1-i2v-turbo":
            d = _as_int(p.get("duration"), 5)
            params["duration"] = d if d in {3, 4, 5} else 5
            if d not in {3, 4, 5}:
                notes.append(f"{model} 时长仅支持 3/4/5 秒，已改为 5 秒")

        if "wan2.6" in model and p.get("shot_type") in {"single", "multi"}:
            params["shot_type"] = p.get("shot_type")
        if model in {"wan2.6-i2v-flash", "wan2.6-i2v", "wan2.5-i2v-preview"} and p.get("audio") is not None:
            params["audio"] = bool(p.get("audio"))
        seed = _valid_seed(p.get("seed"))
        if seed is not None:
            params["seed"] = seed

    if p.get("prompt_extend") is not None:
        params["prompt_extend"] = bool(p.get("prompt_extend"))
    if p.get("watermark") is not None:
        params["watermark"] = bool(p.get("watermark"))
    return params, notes


def normalize_kf2v_parameters(model: str, p: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if is_wan3(model):
        return normalize_i2v_parameters(model, p)
    if model.startswith("wan2.7-i2v"):
        return normalize_i2v_parameters(model, p)

    notes: list[str] = []
    resolution = str(p.get("resolution") or "720P").upper()
    if model == "wanx2.1-kf2v-plus":
        if resolution != "720P":
            notes.append(f"{model} 仅支持 720P，已改为 720P")
        resolution = "720P"
    elif model == "wan2.2-kf2v-flash" and resolution not in {"480P", "720P", "1080P"}:
        resolution = "720P"
        notes.append(f"{model} 分辨率已改为 720P")
    params: dict[str, Any] = {"resolution": resolution, "duration": 5}
    if p.get("prompt_extend") is not None:
        params["prompt_extend"] = bool(p.get("prompt_extend"))
    if p.get("watermark") is not None:
        params["watermark"] = bool(p.get("watermark"))
    seed = _valid_seed(p.get("seed"))
    if seed is not None:
        params["seed"] = seed
    return params, notes


def normalize_happyhorse_parameters(model: str, p: dict[str, Any], with_ratio: bool) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    allowed_resolutions = {"720P", "1080P"} if model.startswith("happyhorse-1.0") else {"480P", "720P", "1080P"}
    resolution = str(p.get("resolution") or "1080P").upper()
    if resolution not in allowed_resolutions:
        resolution = "1080P"
        notes.append(f"{model} 不支持该分辨率，已改为 1080P")
    duration = min(15, max(3, _as_int(p.get("duration"), 5)))
    params: dict[str, Any] = {"resolution": resolution, "duration": duration}
    if with_ratio:
        ratio = str(p.get("ratio") or "16:9")
        if ratio not in HAPPYHORSE_RATIOS:
            ratio = "16:9"
            notes.append(f"{model} 宽高比已改为 16:9")
        params["ratio"] = ratio
    if p.get("watermark") is not None:
        params["watermark"] = bool(p.get("watermark"))
    seed = _valid_seed(p.get("seed"))
    if seed is not None:
        params["seed"] = seed
    return params, notes
