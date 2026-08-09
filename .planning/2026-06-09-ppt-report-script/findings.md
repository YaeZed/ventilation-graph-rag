# Findings: PPT Report Script

## Source Files

- `张明俊-18342379391.pptx`: provided presentation deck.
- `辨识流程视频.mp4`: provided supporting process video. Use only if needed to explain a demo/process slide.

## Extraction Findings

- Structured extraction succeeded.
- Deck has 19 slides.
- Main title: 数据与法规融合的煤矿通风系统安全隐患智能辨识方法研究.
- Speaker metadata: 张明俊; 高科教授; 安全科学与工程; 2026.6.
- Deck structure:
  - Slides 1-2: title and outline.
  - Slides 3-4: problem and goal.
  - Slides 5-6: system positioning and technical architecture.
  - Slides 7-8: regulation knowledge layer, retrieval reasoning, multimodal recognition.
  - Slides 9-17: staged system demonstrations.
  - Slides 18-19: stage summary and closing.
- Existing PowerPoint notes were found on slides 18 and 19 only.

## Visual Findings

- Full slide rendering failed because no supported renderer was available locally.
- Embedded media extraction succeeded: 30 records, including repeated header/logo images, large architecture/screenshot images, and one embedded MP4 on slide 17.
- Key inspected images:
  - Slide 6: technical architecture diagram with interaction layer, business logic layer, data resource layer, and intelligent core layer.
  - Slide 9/10: main chat page showing regulation-backed wind-speed answer and Markdown-style report.
  - Slide 11-13: multi-image recognition workflow, including standby switching test evidence and generated safety analysis.
  - Slide 14-16: image plus sensor fusion, including CO, wind speed, methane, independent ventilation state, and regulation-based risk judgment.
  - Slide 17: embedded video exists, but no local video frame extraction dependency was available.

## Drafting Notes

- Output language selected: Chinese, matching the deck.
- Default speaking context: about 10 minutes, graduate-stage conference/report style.
- The script should emphasize the product/research chain: scattered field evidence -> regulation knowledge structuring -> multimodal evidence recognition -> GraphRAG retrieval -> traceable report.
- Avoid overstating current maturity. Slide 18 says the system has entered an interactive verification stage and next work is comparative experiments, not a fully validated production system.
