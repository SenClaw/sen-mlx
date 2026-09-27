# Tách `local_model` ra Space App — Nghiên cứu

> **Trạng thái: NGHIÊN CỨU, chưa có dòng code nào.** Tài liệu này đo đúng cái
> gì đang dính vào cái gì trong repo hôm nay (2026-08-16), rồi đề xuất một
> đường đi ba pha. Mọi con số LOC/kích thước bên dưới là đếm thật, không ước
> lượng.

Ba việc người dùng đặt ra:

1. Bổ sung SDK để một Space App **đăng ký được LLM provider / model handler**.
2. Migrate `mlx_lm` ra app.
3. Migrate `candle_models` ra app.

Kết luận ngắn: **(1) rẻ và nên làm trước** — nó là hợp đồng, làm xong mới có
chỗ để (2) hạ cánh. **(2) đắt hơn nhiều so với vẻ ngoài**, vì `local-mlx`
không chỉ phục vụ LLM. **(3) nên cân nhắc xoá thay vì migrate.**

---

## 0. Tóm tắt

| | |
|---|---|
| Tổng `src/local_model/` | **30 121 LOC** |
| Phần MLX (`mlx_lm/` + `mlx_native.rs` + `mlx_lm_utils/` + `mlx_prompt.rs` + `chat_template_openai.rs` + `image_input.rs`) | **~22 500 LOC** |
| Phần candle LLM (`candle_engine.rs` + `candle_models/` + `tokenizer_utils/`) | **~2 800 LOC** |
| Phần trung lập, dùng chung (`stream_parser.rs` + `thinking_parse.rs` + `models.rs` + `runtime.rs`) | **~1 900 LOC** |
| Registry + REST + downloader (`gateway/ui_server/local_models.rs`) | **1 532 LOC** |
| Số file ngoài `src/local_model/` tham chiếu vào nó | **15** |
| Số symbol công khai bị dùng từ ngoài | **19** |
| Weights đã tồn tại trên máy dev | **25 GB** ở `~/.senclaw/local-models/` |

Bề mặt ghép nối **nhỏ bất ngờ** — 19 symbol. Cái đắt không phải là gỡ dính,
mà là bốn thứ ở §2.

---

## 1. Hiện trạng: ai gọi vào `local_model`

`grep -rn "local_model" src --include="*.rs"` bên ngoài chính module:

| File | Dùng gì | Sau khi tách thì sao |
|---|---|---|
| `gateway/ui_server/local_models.rs` (36 ref) | `MlxNativeEngine`, `CandleEngine` — registry sống, guard idle-unload, downloader HF, `settings.json` | **Chuyển sang app**, gần như nguyên khối |
| `zen_core/query_llm.rs` (20 ref) | `query_local_mlx` / `query_local_candle_native`, `stream_parser::parse_complete*` | Thay bằng adapter `openai` trỏ vào app (§3) |
| `gateway/ui_server/core.rs` (20 ref) | 14 route `/api/local-models/*` | Chuyển sang app; daemon giữ proxy |
| `config.rs` (8 ref) | `paths.local_models_dir` | **Giữ** — app đọc lại qua env |
| `memory/cognitive/llm_local_mlx.rs` / `llm_local_candle.rs` | `LocalModelRuntime` | Đi qua provider registry như mọi provider khác |
| `gateway/ui_server/whisper.rs` (5) | `WhisperEngine` | Đi cùng MLX (§2b) |
| `gateway/ui_server/ocr.rs` (4) | `OcrEngine` | **Không đi** — `ocr-paddle` là MNN, không dính MLX |
| `memory/embedding.rs` (4) | candle BERT (`local-embed`) | **Không đi** — feature riêng, cross-platform, nhỏ |
| `gateway/ui_server/hf_validate.rs` (3) | `mlx_native::detect_architecture`, `mlx_lm::models::whisper::ModelDimensions` | Bẫy — xem §2e |
| `tts/mms_vits/mod.rs` (2) | `mlx_serial_lock` / `mlx_serial_try_lock` | Bẫy — xem §2a |
| `db/schema.rs` (1) | bảng | Giữ |

---

## 2. Bốn thứ làm việc này đắt hơn vẻ ngoài

### 2a. `mlx_serial_lock` là khoá **toàn tiến trình**, và TTS đang giữ nó

`src/local_model/mlx_native.rs:255` khai `static MLX_SERIAL: Mutex<()>`. Comment
của chính nó nói rõ lý do: *"concurrent MLX work on separate threads corrupts
Metal state"*. Và `src/tts/mms_vits/mod.rs:258,297` đang lấy đúng khoá đó —
MMS-VITS **là một backend MLX** (nó gọi `mlx_sys::mlx_clear_cache` ở dòng 245).

Nên trong daemon hôm nay có **năm** hộ tiêu thụ MLX chia nhau một khoá:

```
MLX_SERIAL
├── MlxNativeEngine        (LLM)           local-mlx
├── WhisperEngine          (ASR)           local-mlx-whisper
├── ZipVoice               (TTS)           local-mlx-tts
├── MmsVitsBackend         (TTS)           local-mlx-tts
└── cognitive MLX embedder                 cognitive-mlx-embed
```

Tách **một** hộ ra khỏi tiến trình không sửa gì cả: bốn hộ còn lại vẫn cần
mlx-rs biên dịch vào daemon → **thời gian build và kích thước binary không
giảm một byte nào**. Đây là phát hiện quan trọng nhất của tài liệu này.

Ngược lại, khi cả năm cùng ra thì khoá này biến mất một cách tự nhiên: mỗi
tiến trình có Metal command queue riêng, mà cái crash AGX đã ghi nhận
(`candle-metal-concurrency-crash`) là lỗi **trong cùng tiến trình**. Điều này
**phải đo, không được giả định** — hai tiến trình cùng đập Metal là một cấu
hình repo chưa từng chạy.

### 2b. Đồ thị feature bị buộc chặt

```
local-mlx-whisper = ["local-mlx", "whisper-audio"]
local-mlx-tts     = ["local-mlx", "whisper-audio"]
cognitive-mlx-embed = ["dep:mlx-rs", "dep:mlx-sys", ...]
DAEMON_FEATURES   = local-mlx,local-embed-metal,local-embed,
                    local-mlx-whisper,local-mlx-tts,ocr-paddle-metal,tts-vieneu
```

Suy ra trực tiếp từ §2a: **đơn vị migrate không phải `mlx_lm`, mà là toàn bộ
runtime MLX** — LLM + ASR + hai TTS + embedder. Gọi nó là `apps/mlx-runtime`.
Một app "chỉ LLM" là công sức bỏ ra mà không thu được lợi ích đã hứa.

### 2c. 25 GB weights đã nằm sẵn ngoài thư mục app

`~/.senclaw/local-models/` đang là 25 GB trên máy dev (10 model + `hf-cache`).
App **không được** dùng `~/.senclaw/space-app-data/mlx-runtime/` — như thế là
bắt mọi người tải lại 25 GB. Daemon phải tiêm `SENCLAW_LOCAL_MODELS_DIR` vào
env của app, và sandbox từng app phải cấp thư mục đó (nhớ bẫy đã ghi trong
`docs/space-app-sandbox.md`: cấp thư mục con không đủ nếu thư mục **cha** bị
chặn).

### 2d. Session app dừng sau 60s, nạp lại weights mất hàng giây

Mặc định `runtime.mode: session` + `idleTimeoutSecs: 60` sẽ biến mỗi lượt chat
cách nhau 2 phút thành một lần nạp lại ~4 GB. Daemon hôm nay đã có đúng bài
toán này và trả lời là `DEFAULT_IDLE_UNLOAD_SECS = 300`
(`local_models.rs:704`). App nên khai `session` với `idleTimeoutSecs` bằng
đúng giá trị người dùng đã đặt trong `settings.json` — tiến trình thoát thì
Metal buffer được trả về OS, tương đương `unload()` mà không cần reaper.

**Không** khai `background`: thế là giữ 4 GB thường trực từ lúc boot, đúng cái
mà `docs/space-app-lifecycle.md` sinh ra để giết.

### 2e. `hf_validate.rs` là bản sao chép của bảng kiến trúc

Doc comment của nó tự nhận: *"Local-LLM rule — mirrors `mlx_native::detect_architecture`"*.
Daemon dùng nó để từ chối một repo HF **trước khi** tải 8 GB. Sau khi tách,
hoặc route validate đi theo app (nhưng thế thì phải khởi động app chỉ để
validate), hoặc bảng arch thành một crate dùng chung. Xem §4 Pha 0.

### 2f. `mlx.metallib` phải nằm cạnh executable

`.github/workflows/desktop.yml:262-272` đang copy `mlx.metallib` vào
`Contents/Resources/` cạnh binary daemon. App có binary riêng → **bundle của
app phải mang metallib của chính nó**, nếu không MLX im lặng không chạy. Đây
là bẫy đóng gói, không phải bẫy code.

---

## 3. SDK: đăng ký LLM provider từ Space App

### Hiện trạng

Không có cơ chế đăng ký động nào. `query_llm::resolve_adapter`
(`src/zen_core/query_llm.rs:319`) là một `match` chuỗi cứng, và
`src/providers/mod.rs` chỉ là **catalog preset tĩnh** (`base_url` + `adapt`),
không phải registry.

### Thiết kế A — khai trong manifest, nói OpenAI wire (**đề xuất**)

Điểm mấu chốt: app **đã có sẵn** `stream_openai_to_channel` và
`chat_template_openai.rs`. Nếu app phát SSE đúng chuẩn OpenAI thì daemon
**không cần adapter mới nào cả** — nó dùng lại adapter `openai` đang có.

Thêm vào `senclaw-manifest.json`:

```json
"llm": {
  "providers": [{
    "id": "mlx-local",
    "displayName": "MLX (Apple Silicon)",
    "path": "/v1",
    "adapt": "openai",
    "modelsPath": "/v1/models",
    "capabilities": { "vision": true, "tools": true, "streaming": true }
  }]
}
```

Hành vi daemon **soi gương y hệt `mcp.autoRegister`** — đây là lý do thiết kế
này rẻ, mọi mảnh đã tồn tại:

| MCP hôm nay | LLM provider tương ứng |
|---|---|
| Đăng ký vào registry MCP | bảng `space_app_llm_providers` |
| Session app trỏ vào **app proxy** `/api/space/apps/<id>/proxy<mcp.path>` | trỏ vào `/api/space/apps/<id>/proxy/v1` → gọi lần đầu thì spawn app |
| Tool list cache ở `<app>/.senclaw/mcp-tools.json` | model list cache ở `<app>/.senclaw/llm-models.json` → model picker vẫn thấy model khi app đang tắt |
| Proxy đóng dấu `SENCLAW_TOKEN_ACCESS_APP` | y hệt |
| Sai chính tả `mode` → im lặng fallback | validator manifest phải bắt, xem dưới |

Bề mặt sửa trong daemon thực sự nhỏ: một bảng, một registry, một route proxy,
và **một nhánh** thêm vào `resolve_adapter`.

### Thiết kế B — RPC qua dispatch protocol

Dùng `app-space-sdk/src/dispatch` với capability `llm.stream`. Nhiều việc hơn,
phải viết lại streaming, không được gì. **Loại.**

### Phần SDK phải viết (bốn ngôn ngữ, Rust trước)

```rust
// app-space-sdk/src/llm.rs
pub trait LlmProvider: Send + Sync {
    fn models(&self) -> Vec<ModelCard>;
    async fn chat(&self, req: ChatRequest, sink: ChunkSink) -> Result<()>;
}
/// Dựng sẵn axum router: /v1/chat/completions (SSE + non-stream), /v1/models.
pub fn openai_router<P: LlmProvider + 'static>(p: Arc<P>) -> axum::Router;
```

Tác giả app chỉ `impl LlmProvider`, còn wire format là của SDK. Kèm theo:

- `manifest::LlmDecl` + validator. Giữ đúng kỷ luật "sai chính tả phải kêu"
  đã có với `runtime.mode`: `adapt` không thuộc tập adapter daemon biết → **từ
  chối**, chứ không im lặng fallback về `openai`.
- Tự ghi `.senclaw/llm-models.json` lúc khởi động.
- Bản Node/Python/Go làm sau, cùng validator.

---

## 4. Kế hoạch ba pha

### Pha 0 — gỡ dính tại chỗ, **chưa có app nào**

Tách phần trung lập ra crate workspace `senclaw-local-core` (không dep MLX,
không dep candle):

- `stream_parser.rs` (1 506 LOC) — daemon vẫn cần cho candle path & test
- `thinking_parse.rs` (122)
- `models.rs` (205) — `KNOWN_MODELS`, `infer_vision_from_id`
- bảng arch mà `hf_validate.rs` đang chép tay (§2e)

Daemon và app tương lai cùng dep vào nó. Làm bước này **trước** thì bài toán
trùng lặp 1 500 LOC không bao giờ phát sinh.

### Pha 1 — SDK + provider registry (Thiết kế A)

Ship kèm một app demo tầm thường (`echo` provider, ~200 LOC). Wire được test
thật trước khi 22 500 LOC dọn nhà.

### Pha 2 — `apps/mlx-runtime`

Chuyển sang app:

- `mlx_lm/`, `mlx_lm_utils/`, `mlx_native.rs`, `mlx_prompt.rs`,
  `chat_template_openai.rs`, `image_input.rs`
- `whisper_transcribe.rs` + `audio.rs` (ASR)
- ZipVoice + `tts/mms_vits/` (TTS)
- cognitive MLX embedder
- `local_models.rs` — registry, downloader HF, `settings.json`, 14 route REST

App phơi ra: `/v1/chat/completions`, `/v1/audio/transcriptions`,
`/v1/embeddings` (đều OpenAI-shaped) + vài tool MCP quản lý model.

Manifest: `requires.os: ["macos"]` + arch arm64 — **đây là một cái được thật**:
hôm nay việc "máy này không chạy được MLX" là một cờ biên dịch; sau khi tách nó
thành một câu từ chối lúc cài.

Daemon giữ lại: `senclaw-local-core`, `LocalModelRuntime` health, `hf_validate`
(mỏng đi), OCR, `local-embed` (candle BERT), `tts-vieneu` (ONNX).

### Pha 3 — candle: **cân nhắc xoá thay vì migrate**

`local-candle` LLM là ~2 800 LOC cho 7–12 tok/s, so với MLX 60–100 tok/s trên
cùng máy (bảng trong `src/local_model/mod.rs`). Trên macOS nó bị MLX áp đảo;
ngoài macOS thì Ollama đã phủ. Migrate nó là bê một backend không ai chọn sang
một tiến trình mới.

Đề xuất: **xoá `local-candle`**, giữ nguyên `local-embed` (candle BERT cho
`memory/embedding.rs` — feature khác, nhỏ, cross-platform, đang được dùng).

Nếu vẫn muốn giữ, nó là app dễ nhất trong ba app — thuần Rust, cross-platform,
không dính khoá Metal nào.

---

## 5. Rủi ro còn mở

| Rủi ro | Mức | Ghi chú |
|---|---|---|
| Hai tiến trình cùng dùng Metal | **Cao** | Chưa từng chạy trong repo. Phải đo trước Pha 2, không giả định |
| Prefix cache / KV mất khi app restart | Thấp | Hôm nay daemon restart cũng mất y hệt — trung lập |
| Overhead SSE so với `mpsc<String>` in-process | Thấp | Decode 60–100 tok/s ≈ 1 chunk / 10–16 ms; loopback SSE dư sức |
| App bị `sandbox` chặn khỏi thư mục 25 GB | Trung bình | §2c — cả bẫy thư mục cha |
| Thiếu `mlx.metallib` trong bundle app | Trung bình | §2f — hỏng im lặng |
| `hf_validate` lệch pha với bảng arch trong app | Trung bình | Pha 0 xử lý |

---

## 6. Việc nên làm ngay

1. **Pha 0** — dựng `senclaw-local-core`. Không phụ thuộc quyết định nào ở
   trên, và làm sạch cho mọi hướng đi.
2. **Đo Metal đa tiến trình** — chạy hai tiến trình cùng nạp MLX và sinh token,
   xem có AGX assert không. Kết quả này quyết định Pha 2 có khả thi không.
3. **Pha 1** — SDK + registry, với app demo.

Chỉ sau khi (2) xanh thì Pha 2 mới đáng bắt đầu.
