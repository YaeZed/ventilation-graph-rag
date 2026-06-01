# Plan: 传感器数据接入 + 多图片联合辨识

## Implementation Status

Completed on 2026-06-01.

- Frontend: added `SensorInputPanel.vue`, `SensorDataBadge.vue`, `MultiImageBar.vue`, shared multimodal types, multi-image draft queues, sensor data draft state, message rendering, persistence, search, and PDF export support.
- API: `POST /api/chat/` accepts JSON `sensor_data`; `POST /api/chat/upload/` and `/api/chat/stream/` accept repeated multipart `images` plus `sensor_data`, while preserving legacy single `image`.
- RAG: `VentilationRAGPipeline.query()` now accepts `image_paths` and `sensor_data`; single-image, multi-image, sensor-only, and fused flows route explicitly.
- Vision: `VentilationVisionExtractor.extract_multi()` performs per-image observation, merged concept retrieval, and multi-image joint analysis with per-image observations and cross-image findings.
- Generation: `_build_multimodal_prompt()` and `generate_multimodal_answer(_stream)` produce image/data/regulation cross-validation reports.
- UI polish: sensor type selection reuses `SettingsSelect`; the multi-image add button stays square after thumbnails render; sent user-message images render immediately, can be opened in an in-page centered preview, and survive refresh through compressed local previews plus backend attachment retry.
- Validation: Python compile check, `web_backend/manage.py check`, Django chat multimodal fake-pipeline smoke, `vue-tsc --build`, and `vite build` passed.

## Context

本计划实施前，系统仅处理单张图片。论文题目为《数据与法规融合的煤矿通风系统安全隐患智能辨识方法研究》，"数据"指传感器时序/数值数据，"融合"包含多模态异构数据源在语义层面的交叉验证。

当前缺少两个核心能力：
1. **传感器数据接入** — 风速、瓦斯浓度、温度等数值监测数据未进入辨识链路
2. **多图片联合辨识** — 单次只能分析一张图片，无法交叉关联同一工作面的多角度信息

## 传感器数据接入

### 交互设计

在对话输入栏左侧，图片上传按钮旁增加一个"传感器数据"按钮：

```
[图片+按钮] [数据+按钮] [输入框] [发送按钮]
```

点击弹出传感器数据输入面板（inline panel / modal），支持两种输入方式：

**方式 1：手动输入（默认）**
```
┌─ 传感器数据 ──────────────────────┐
│ 数据类型    □ 风速(m/s)  □ 瓦斯浓度(%) │
│            □ CO浓度(ppm) □ 温度(℃)    │
│            □ 氧气浓度(%) □ 其他        │
│                                     │
│ 检测地点    [掘进工作面        ▼]    │
│ 风速        0.12       m/s          │
│ 瓦斯浓度    0.08       %            │
│                                     │
│ [+ 添加数据项]                       │
│ [确认]  [取消]                       │
└─────────────────────────────────────┘
```

**方式 2：CSV 粘贴/上传**
```
粘贴 CSV 或拖拽文件：
时间, 风速(m/s), 瓦斯(%), 温度(℃)
08:00, 0.25, 0.05, 22.1
08:05, 0.23, 0.06, 22.3
08:10, 0.12, 0.08, 22.5    ← 异常点
```

### 数据结构

```typescript
interface SensorEntry {
  type: 'wind_speed' | 'methane' | 'co' | 'temperature' | 'oxygen' | 'custom'
  label: string          // 显示名
  value: number
  unit: string           // m/s, %, ppm, ℃
  location?: string      // 检测地点
  timestamp?: string     // 数据时间
  thresholdRef?: string  // 关联的规程阈值出处
}

interface SensorData {
  entries: SensorEntry[]
  location: string       // 全局检测地点
  source: 'manual' | 'csv'
  rawCsv?: string
}
```

### 后端数据流

```
图片 → Pass 1 VL观察 → 不确定概念检索 → Pass 2 VL分析
                                               │
传感器数据 ─────────────────────────────────────┘
                                               │
                    ┌──────────────────────────┘
                    ▼
           汇编为统一上下文
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  Cypher 模板检索          向量兜底检索
  (数值可精确绑定)        (语义描述兜底)
        │                       │
        └───────────┬───────────┘
                    ▼
            LLM 生成辨识报告
            (图片证据 + 数据证据 + 法规依据 → 交叉验证结论)
```

### 生成 prompt 扩展

在现有 `_build_image_prompt` 基础上扩展一个新方法 `_build_multimodal_prompt`：

```
【图片分析结果】
- 识别场景：{scene_name}
- 主要隐患：{primary_hazard}
- 关键观察：{key_observations}

【传感器实测数据】
| 参数 | 数值 | 单位 | 检测地点 |
|------|------|------|---------|
| 风速 | 0.12 | m/s | 掘进工作面 |
| 瓦斯浓度 | 0.08 | % | 掘进工作面 |

【规程数值约束】
| 参数 | 最小值 | 最大值 | 适用地点 | 来源条款 |
|------|--------|--------|---------|---------|
| 最低风速 | 0.25 | - | 掘进中的岩巷 | 第一百五十七条 |

【参考规程内容】
{context}

请进行交叉验证分析：
1. 图片证据与传感器数据是否指向同一隐患？
2. 传感器数值与规程阈值的对比结果
3. 综合图片+数据+规程，给出最终判定

输出格式参考（自由发挥）：
- **交叉验证分析**：图片与数据的对应关系
- **数据合规性**：逐项比对传感器数值与规程阈值
- **综合结论**：融合图-文-数的最终判定
- **规程依据**：引用条款和数值
- **整改建议**：可操作措施
- **补充观察**
```

### 修改文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `frontend/src/components/SensorInputPanel.vue` | **新建** | 传感器数据输入面板（手动+CSV） |
| `frontend/src/components/SensorDataBadge.vue` | **新建** | 消息卡片中的传感器数据展示 |
| `frontend/src/stores/chat.ts` | 改造 | message 增加 sensorData 字段 |
| `frontend/src/views/HomeView.vue` | 改造 | 输入栏增加传感器按钮，提交传递 sensorData |
| `agent/rag_system/ventilation_generation.py` | 新增 | `_build_multimodal_prompt` / `generate_multimodal_answer_stream` |
| `agent/rag_system/ventilation_rag_pipeline.py` | 改造 | `query()` 增加 sensor_data 参数，生成步骤事件增加传感器比对 |
| `web_backend/chat/views.py` | 微调 | SSE 流式透传 sensor_data |

---

## 多图片联合辨识

### 交互设计

支持连续选择多张图片（同一对话中），图片以缩略图队列形式展示在输入栏上方：

```
┌─────────────────────────────────────┐
│ [图1缩略图×] [图2缩略图×] [图3缩略图×] │
│ [+ 添加图片]                         │
│                                     │
│ 补充描述：检查掘进工作面整体通风状况…  │
│                                     │
│ [数据+] [图片+] [输入框] [发送]      │
└─────────────────────────────────────┘
```

### 后端数据流

```
图片1、图片2、图片3 ──→ 各自独立 Pass 1 VL观察
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   各图的不确定概念合并      共同场景猜测
        │                       │
        ▼                       ▼
   合并去重 → 概念检索    场景协商（多图投票）
        │                       │
        └───────────┬───────────┘
                    ▼
         Pass 2: 多图综合分析 VL
         (所有图片 + 所有概念定义 + 交叉关联提示)
                    │
                    ▼
              统一结构化输出
         (scene + 汇总的 structured_fields
              + 每图独立的 key_observations
              + 综合 primary_hazard)
                    │
                    ▼
          Cypher检索 → 向量兜底 → 生成
```

### VL 多图分析 prompt 设计

Pass 1 对每张图独立调用（温度 0.3）：
```
你正在检查一个煤矿井下工作面的多张现场照片。
这是第 {index}/{total} 张照片。

请观察图片，回答：
1. 你看到了哪些通风设施、设备和环境特征？
2. 可能的安全隐患？需要确认的通风概念？
3. 与其他照片可能有什么关联？（如果有的话）
```

Pass 2 多图综合分析（温度 0.25）：
```
你已获得以下信息：

【图片 1 观察】
- 观察：{obs1}
- 不确定概念：{concepts1}

【图片 2 观察】
- 观察：{obs2}
- 不确定概念：{concepts2}

【图片 3 观察】
- 观察：{obs3}
- 不确定概念：{concepts3}

【概念参考卡片】
{merged_concept_cards}

现在综合所有图片和概念定义，进行交叉关联分析：
1. 各图之间的空间关系和因果关系
2. 联合证据指向的共同隐患
3. 单独某张图无法判断但多图联合可以确认的问题

请选择最匹配的场景，提取结构化字段，输出综合判定。
```

### VisionExtractionResult 扩展

```python
@dataclass
class MultiImageResult:
    scene_id: str
    scene_name: str
    structured_fields: Dict[str, Any]
    description: str
    confidence: float
    primary_hazard: str
    risk_level: str
    key_observations: List[str]            # 综合观察
    per_image_observations: Dict[int, str] # 每图的独立观察
    cross_image_findings: List[str]        # 跨图关联发现
    uncertain_concepts: List[str]          # 合并去重
    concepts_retrieved: List[Dict]
```

### 修改文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `frontend/src/components/MultiImageBar.vue` | **新建** | 多图片缩略图队列 |
| `frontend/src/views/HomeView.vue` | 改造 | 支持多文件选择，提交传递多个 image_path |
| `frontend/src/stores/chat.ts` | 改造 | message 增加 images[] 字段 |
| `agent/rag_system/ventilation_vision_extractor.py` | 新增 | `extract_multi()` 方法 |
| `agent/rag_system/ventilation_rag_pipeline.py` | 新增 | `_query_with_multi_image_stream()` |
| `web_backend/chat/views.py` | 微调 | 支持多文件上传 |

---

## 实施顺序

```
Phase 1: 传感器数据接入（独立，不依赖多图改造）
    ↓
Phase 2: 多图片联合辨识（独立于传感器，但两者最终汇合）
    ↓
Phase 3: 双模态融合 prompt（传感器 + 多图交叉验证）
```

Phase 3 是最终形态——一张图 + 一个传感器数值 + 多张图 = 完整的"数据与法规融合"演示。

## 验证方式

**传感器数据**：
- 上传一张风机图片，手动输入风速 0.12 m/s，预期报告包含传感器数值与规程阈值对照
- 粘贴 CSV 数据，预期解析出结构化条目并注入 prompt

**多图片**：
- 上传同一掘进面的 2-3 张不同角度照片，预期报告提到跨图关联发现
- 单张图不能判定但多图联合确认的 case

**双模态融合**：
- 图片 + 传感器数据一同提交，预期报告包含交叉验证分析章节
