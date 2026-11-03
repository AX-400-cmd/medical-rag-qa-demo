# 医疗知识库 RAG 问答 Demo（Medical RAG QA）

> 一个**面向医疗健康场景**的检索增强生成（RAG）问答服务。重点不在堆模型，而在把
> **“FastAPI 工程化 + 向量检索 + 重排序 + 结构化输出 + 医疗安全边界”** 这条完整链路
> 做扎实、做可解释、可离线跑通——这正是医疗 AI Agent 岗位最核心的落地能力。

## 一、它解决什么问题 / 对应岗位能力

| 本项目能力 | 对应「医疗 AI Agent 应用工程师」JD 要求 |
| --- | --- |
| FastAPI 提供 REST 接口、Schema 校验 | **FastAPI（JD 加分项）** |
| 每次回答都是强类型 `AskResponse`（含置信度/来源/安全标记） | **结构化输出与校验** |
| 紧急/诊断类问题强制 `needs_human_review` | **人工接管 / 可追溯** |
| 命中安全规则即转人工、附免责声明、对隐私脱敏 | **医疗场景严格边界** |
| 检索片段全部带 `sources` 溯源 | **可追溯** |

## 二、系统架构

```
         ┌────────────┐
  提问 → │  /ask 接口  │  FastAPI + Pydantic 校验
         └─────┬──────┘
               │
       ┌───────▼────────┐
       │ 向量检索 top-k  │  本地 TF-IDF / 可换稠密 embedding(BGE等)
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │   重排序 Rerank │  LLM pointwise 打分 / 分数兜底
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │ 结构化生成      │  LLM(JSON) / 离线接地回答
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │ 医疗安全边界    │  紧急/诊断/低置信 → 转人工 + 脱敏 + 免责
       └───────┬────────┘
               │
         └──► 结构化响应（含 sources / safety_flags）
```

## 三、怎么跑（离线，零 API Key）

```bash
cd 能力补强/医疗RAG实战
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# 打开 http://127.0.0.1:8000/docs 可直接调试
```

调用示例：

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"感冒和流感有什么区别","top_k":3}'
```

自测（离线管线，不启服务）：

```bash
py -3.12 tests/test_pipeline.py
```

## 四、接入真实大模型 / 稠密向量（可选）

配置环境变量即可切换，无需改代码：

```bash
# 生成与重排用的大模型（兼容 OpenAI 接口）
set LLM_BASE_URL=https://你的服务/v1/chat/completions
set LLM_API_KEY=sk-xxxx
set LLM_MODEL=你的模型名

# 稠密向量（不配则用本地 TF-IDF 稀疏向量）
set EMBED_BASE_URL=https://你的服务/v1/embeddings
set EMBED_API_KEY=sk-xxxx
set EMBED_MODEL=text-embedding-3-small
```

## 五、医疗安全设计（项目重点）

- **紧急危重识别**：命中胸痛/呼吸困难/昏迷/中毒等关键词 → 强制转人工 + 建议 120，绝不给自行处理建议。
- **诊断/用药边界**：涉及“帮我诊断/开药/我得了什么病”一律不下结论，转执业医师。
- **低置信兜底**：检索置信度低于阈值时不拍板，建议专业核实。
- **隐私脱敏**：对身份证号、手机号、显式姓名做脱敏。
- 所有响应附带统一免责声明，保证**可追溯、可接管**。

> ⚠️ 本项目知识库为**健康科普示例数据**，**不能替代专业医疗诊断**。仅用于工程能力演示。

## 六、面试三分钟故事（填空即可）

> 我做了一套面向医疗场景的 RAG 问答系统，解决“健康问答容易胡说、且不能替用户下医学结论”的问题。
> 链路是：知识库切块 → 向量化检索 top-k（默认本地 TF-IDF，可换 BGE 稠密向量）
> → 重排序精排 → 用 Pydantic 约束大模型做**结构化输出** → 再过一层**医疗安全边界**
> （紧急转人工、诊断不越界、低置信不拍板、隐私脱敏）。
> 我特意权衡了【稀疏 vs 稠密向量】【要不要混合 BM25】【结构化输出怎么防幻觉】，
> 遇到的坑是【片段不带出处会张冠李戴】，用【来源溯源 + 安全标记】解决。
> 上生产我会把 embedding 换 BGE、检索换 FAISS + 混合检索、知识库换成经审核的医学语料。

## 七、开源说明

本项目为可独立运行的工程 Demo。如需发布到 GitHub：

```bash
git init
git add -A
git commit -m "feat: 医疗知识库 RAG 问答 Demo（FastAPI + 检索/重排/安全边界）"
git remote add origin <你的仓库地址>
git push -u origin main
```

推送前请确认 `data/knowledge_base.json` 仅为科普示例数据、不含任何个人隐私。
