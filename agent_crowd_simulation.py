"""Agent 人群模拟研究平台 - Agent Crowd Simulation Platform

A Streamlit app that uses crewAI agents to simulate multi-agent crowd interactions.
Each agent plays a different social role and responds to a user-defined topic.
"""

import threading
import time
from dataclasses import dataclass, field

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Crew, Process, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.task_events import TaskCompletedEvent, TaskStartedEvent


# ---------------------------------------------------------------------------
# Message store: thread-safe shared state between crew thread and Streamlit
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    role: str
    content: str
    msg_type: str
    timestamp: float = field(default_factory=time.time)


class ChatMessageStore:
    """Thread-safe container for chat messages shared between threads."""

    def __init__(self):
        self._messages: list[ChatMessage] = []
        self._lock = threading.Lock()
        self._done = False
        self._error: str | None = None

    def add(self, msg: ChatMessage):
        with self._lock:
            self._messages = [*self._messages, msg]

    def get_all(self) -> list[ChatMessage]:
        with self._lock:
            return list(self._messages)

    def mark_done(self):
        with self._lock:
            self._done = True

    def mark_error(self, error: str):
        with self._lock:
            self._error = error
            self._done = True

    @property
    def done(self) -> bool:
        with self._lock:
            return self._done

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error


# ---------------------------------------------------------------------------
# Module-level active store pointer — handlers write here (registered once)
# ---------------------------------------------------------------------------

_active_store: ChatMessageStore | None = None


# ---------------------------------------------------------------------------
# Register event handlers ONCE at module level (singleton event bus)
# ---------------------------------------------------------------------------

@crewai_event_bus.on(AgentExecutionStartedEvent)
def _on_agent_started(source, event: AgentExecutionStartedEvent):
    store = _active_store
    if store is not None:
        store.add(ChatMessage(
            role=event.agent.role,
            content=f"*{event.agent.role} 开始发言...*",
            msg_type="started",
        ))


@crewai_event_bus.on(AgentExecutionCompletedEvent)
def _on_agent_completed(source, event: AgentExecutionCompletedEvent):
    store = _active_store
    if store is not None:
        store.add(ChatMessage(
            role=event.agent.role,
            content=event.output,
            msg_type="completed",
        ))


@crewai_event_bus.on(TaskStartedEvent)
def _on_task_started(source, event: TaskStartedEvent):
    store = _active_store
    if store is not None:
        desc = ""
        if event.task and hasattr(event.task, "description"):
            desc = event.task.description[:80]
        store.add(ChatMessage(
            role="system",
            content=f"📋 任务开始: {desc}...",
            msg_type="task_started",
        ))


@crewai_event_bus.on(TaskCompletedEvent)
def _on_task_completed(source, event: TaskCompletedEvent):
    store = _active_store
    if store is not None:
        store.add(ChatMessage(
            role="system",
            content="✅ 任务完成",
            msg_type="task_completed",
        ))


# ---------------------------------------------------------------------------
# Avatar mapping for each role
# ---------------------------------------------------------------------------

ROLE_AVATARS = {
    "老板 (Boss)": "👔",
    "资深员工 (Senior)": "🦊",
    "新人 (Newbie)": "🐣",
    "HR": "🎭",
    "分析师 (Analyst)": "📊",
    "system": "⚙️",
}


# ---------------------------------------------------------------------------
# Build the crew: 4 social roles + sequential task chain
# ---------------------------------------------------------------------------

def build_crew(topic: str, num_rounds: int = 3) -> Crew:
    """Create a Crew with 4 social role agents discussing the given topic.

    Generates num_rounds × 4 conversation tasks plus a final analyst task.
    """

    boss = Agent(
        role="老板 (Boss)",
        goal="推动项目按时交付，维护自己的权威",
        backstory=(
            "你是公司部门老板，强势、关注KPI、喜欢甩锅。"
            "你习惯用命令式语气说话，经常把'deadline'挂在嘴边。"
            "当出了问题时，你第一反应是找人背锅而不是解决问题。"
        ),
        verbose=False,
        allow_delegation=False,
    )

    senior = Agent(
        role="资深员工 (Senior)",
        goal="保住自己的地位不被新人取代，同时邀功",
        backstory=(
            "你是公司老油条，工作十年，擅长邀功和暗中使绊。"
            "你表面上对所有人都很客气，但说话总是绵里藏针。"
            "你最擅长把自己的失误推给别人，把别人的功劳揽到自己身上。"
        ),
        verbose=False,
        allow_delegation=False,
    )

    newbie = Agent(
        role="新人 (Newbie)",
        goal="证明自己的能力，获得晋升机会",
        backstory=(
            "你是刚入职半年的新人，充满热情但有些天真。"
            "你总是积极发言想表现自己，但经常不知不觉踩到别人的坑。"
            "你还没学会办公室的潜规则，说话太直接。"
        ),
        verbose=False,
        allow_delegation=False,
    )

    hr = Agent(
        role="HR",
        goal="维持团队表面和谐，收集信息",
        backstory=(
            "你是HR部门的资深员工，擅长打太极和和稀泥。"
            "你表面上谁都不得罪，但私下收集所有人的八卦。"
            "你的发言总是两面讨好，最后各打五十大板。"
        ),
        verbose=False,
        allow_delegation=False,
    )

    analyst = Agent(
        role="分析师 (Analyst)",
        goal="从组织行为学角度，用学术研究框架分析多智能体群体互动动态",
        backstory=(
            "你是组织行为学与人力资源管理领域的研究者，"
            "熟悉 Kacmar & Ferris (1991) 的组织政治感知量表(POPS)、"
            "社会交换理论、资源保存理论(COR)、印象管理理论、"
            "Mintzberg 的权力博弈框架和 Pfeffer 的资源依赖理论。"
            "你擅长从对话数据中识别自变量、因变量、中介变量和调节变量，"
            "并将观察到的行为映射到经管论文常用的硬指标（绩效类）和软指标（行为类）。"
        ),
        verbose=False,
        allow_delegation=False,
    )

    # Dynamic task chain: num_rounds × 4 agents per round
    agents = [boss, senior, newbie, hr]

    # Per-role prompt templates: (round 1, round 2+)
    round1_prompts = {
        "老板 (Boss)": (
            f"会议议题：{topic}\n"
            "你是老板，请宣布这个议题并表达你的立场。"
            "用你一贯的强势风格发言，可以暗示要追究责任。"
            "用中文回答。"
        ),
        "资深员工 (Senior)": (
            f"会议议题：{topic}\n"
            "你是资深员工，请回应老板的发言。"
            "表面上附和老板，但暗中把责任往新人身上引导。"
            "用中文回答。"
        ),
        "新人 (Newbie)": (
            f"会议议题：{topic}\n"
            "你是新人，请回应前面的讨论。"
            "你想表现自己但不太懂办公室政治，"
            "可能会天真地说出一些让自己陷入困境的话。"
            "用中文回答。"
        ),
        "HR": (
            f"会议议题：{topic}\n"
            "你是HR，请对前面所有人的发言做总结。"
            "用你擅长的和稀泥方式，各打五十大板，"
            "表面上化解矛盾但实际上什么都没解决。"
            "用中文回答。"
        ),
    }

    followup_prompts = {
        "老板 (Boss)": (
            f"会议议题：{topic}\n"
            "这是第{{round}}轮讨论。根据前面的讨论继续推进，"
            "你可以追问、施压或甩锅。保持你的强势风格。"
            "用中文回答。"
        ),
        "资深员工 (Senior)": (
            f"会议议题：{topic}\n"
            "这是第{{round}}轮讨论。根据局势变化调整策略，"
            "可以见风使舵、邀功或继续给新人挖坑。"
            "用中文回答。"
        ),
        "新人 (Newbie)": (
            f"会议议题：{topic}\n"
            "这是第{{round}}轮讨论。根据前面的讨论回应，"
            "你可能开始意识到被针对，尝试辩解或反击。"
            "用中文回答。"
        ),
        "HR": (
            f"会议议题：{topic}\n"
            "这是第{{round}}轮讨论。继续观察局势，适时调停，"
            "但也在暗中收集信息，为后续做准备。"
            "用中文回答。"
        ),
    }

    expected_outputs = {
        "老板 (Boss)": "老板的发言",
        "资深员工 (Senior)": "资深员工的回应",
        "新人 (Newbie)": "新人的回应",
        "HR": "HR的回应",
    }

    tasks: list[Task] = []

    for round_idx in range(num_rounds):
        round_num = round_idx + 1
        for agent in agents:
            if round_idx == 0:
                desc = round1_prompts[agent.role]
            else:
                desc = followup_prompts[agent.role].format(round=round_num)

            # Context: up to the last 2 rounds (8 tasks)
            ctx = tasks[-8:] if tasks else []

            task = Task(
                description=desc,
                expected_output=f"第{round_num}轮 - {expected_outputs[agent.role]}",
                agent=agent,
                context=ctx,
            )
            tasks.append(task)

    # Final analyst task — context is ALL conversation tasks
    analyst_task = Task(
        description=(
            f"会议议题：{topic}\n"
            f"以上是{num_rounds}轮多智能体群体讨论的完整记录。\n"
            "请从组织行为学角度进行深度学术分析，严格按以下4个模块输出：\n\n"
            "## 1. 硬指标：绩效影响分析\n"
            "- **任务绩效(Task Performance)**：各角色行为对工作产出的影响\n"
            "- **组织绩效**：团队整体决策质量、协作效率的变化\n\n"
            "## 2. 软指标：行为变量识别\n"
            "- **组织公民行为(OCB)**：是否有人主动帮助/利他行为，或OCB被抑制\n"
            "- **创新行为(Innovative Behavior)**：新想法是否被鼓励还是被群体互动环境扼杀\n"
            "- **不道德行为/亲组织不道德行为(UPB)**：甩锅、信息操纵、背后中伤等\n"
            "- **离职意愿(Turnover Intention)信号**：哪些角色表现出退缩/脱离迹象\n\n"
            "## 3. 变量关系建模\n"
            "- **自变量(IV)**：核心驱动因素（如组织政治感知(POPS)）\n"
            "- **因变量(DV)**：结果变量（绩效、行为等）\n"
            "- **中介变量(Mediator)**：传导机制（工作满意度、组织承诺、心理压力等）\n"
            "- **调节变量(Moderator)**：边界条件（政治技能、领导风格、道德认同等）\n"
            "- **可提炼的研究假设(H1, H2...)**\n\n"
            "## 4. 量表与方法论建议\n"
            "- 对应的经典量表推荐（POPS、OCB量表、任务绩效量表等）\n"
            "- 适用的理论框架（社会交换理论、COR理论、组织公平理论等）\n"
            "- 建议的研究模型路径图描述\n\n"
            "用中文回答。"
        ),
        expected_output="结构化的多智能体群体互动分析报告",
        agent=analyst,
        context=list(tasks),
    )
    tasks.append(analyst_task)

    return Crew(
        agents=[*agents, analyst],
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Background thread runner
# ---------------------------------------------------------------------------

def run_crew_in_background(crew: Crew, store: ChatMessageStore):
    """Run crew.kickoff() in a background thread."""
    global _active_store
    try:
        _active_store = store
        crew.kickoff()
        store.mark_done()
    except Exception as e:
        store.mark_error(str(e))
    finally:
        _active_store = None


# ---------------------------------------------------------------------------
# Render chat messages in Streamlit
# ---------------------------------------------------------------------------

CONVERSATION_ROLES = {"老板 (Boss)", "资深员工 (Senior)", "新人 (Newbie)", "HR"}


def format_messages_as_markdown(
    messages: list[ChatMessage], topic: str, num_rounds: int,
) -> str:
    """Format finished messages into a Markdown document for export."""
    lines = [
        "# Agent 人群模拟记录",
        f"\n## 话题：{topic}",
        f"\n对话轮数：{num_rounds}",
        "",
    ]

    current_round = 0
    msgs_in_round = 0
    analyst_content = ""

    for msg in messages:
        if msg.msg_type != "completed":
            continue

        if msg.role == "分析师 (Analyst)":
            analyst_content = msg.content
            continue

        if msg.role not in CONVERSATION_ROLES:
            continue

        if msgs_in_round % 4 == 0:
            current_round += 1
            lines.append(f"\n### 第 {current_round} 轮\n")
        msgs_in_round += 1

        avatar = ROLE_AVATARS.get(msg.role, "")
        lines.append(f"**{avatar} {msg.role}:**\n")
        lines.append(f"{msg.content}\n")

    if analyst_content:
        lines.append("\n---\n")
        lines.append("### 分析总结\n")
        lines.append(analyst_content)

    return "\n".join(lines)


def render_messages(messages: list[ChatMessage]):
    """Render conversation messages (excludes analyst output)."""
    for msg in messages:
        if msg.msg_type == "started":
            continue
        if msg.role == "分析师 (Analyst)":
            continue
        avatar = ROLE_AVATARS.get(msg.role, "💬")
        if msg.role == "system":
            st.caption(msg.content)
        else:
            with st.chat_message(msg.role, avatar=avatar):
                st.markdown(f"**{msg.role}**")
                st.write(msg.content)


def render_analyst(messages: list[ChatMessage]):
    """Render the analyst output in a separate expander."""
    for msg in messages:
        if msg.role == "分析师 (Analyst)" and msg.msg_type == "completed":
            with st.expander("📊 群体互动分析", expanded=False):
                st.markdown(msg.content)
            return


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Agent 人群模拟研究",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 Agent 人群模拟研究")
    st.caption("基于 LLM 多智能体的群体行为模拟与分析")

    # -- Sidebar --
    with st.sidebar:
        st.header("⚙️ 设置")
        topic = st.text_input(
            "会议话题",
            value="项目延期了，谁来背锅？",
            help="输入一个模拟场景话题",
        )
        num_rounds = st.slider(
            "对话轮数", min_value=1, max_value=10, value=3,
            help="每轮4个角色各发言一次，轮数越多剧情越丰富",
        )
        start_btn = st.button(
            "🚀 开始模拟", type="primary", use_container_width=True,
        )

        st.divider()
        st.subheader("角色介绍")
        for role, avatar in ROLE_AVATARS.items():
            if role != "system":
                st.write(f"{avatar} **{role}**")

    # -- Initialize session state --
    if "store" not in st.session_state:
        st.session_state.store = None
    if "running" not in st.session_state:
        st.session_state.running = False
    if "finished_messages" not in st.session_state:
        st.session_state.finished_messages = []
    if "sim_topic" not in st.session_state:
        st.session_state.sim_topic = ""
    if "sim_num_rounds" not in st.session_state:
        st.session_state.sim_num_rounds = 3

    # -- Handle start button --
    if start_btn and not st.session_state.running:
        st.session_state.store = ChatMessageStore()
        st.session_state.running = True
        st.session_state.finished_messages = []
        st.session_state.sim_topic = topic
        st.session_state.sim_num_rounds = num_rounds

        crew = build_crew(topic, num_rounds)

        thread = threading.Thread(
            target=run_crew_in_background,
            args=(crew, st.session_state.store),
            daemon=True,
        )
        thread.start()

    # -- Main area: render based on state --
    if st.session_state.running and st.session_state.store is not None:
        store: ChatMessageStore = st.session_state.store

        if not store.done:
            st.info("🔄 模拟进行中，请稍候...")
            render_messages(store.get_all())
            time.sleep(1.0)
            st.rerun()

        # Crew finished — save results and stop polling
        final_messages = store.get_all()
        st.session_state.finished_messages = final_messages
        st.session_state.running = False

        if store.error:
            st.error(f"模拟出错: {store.error}")
        else:
            st.success("✅ 模拟完成！")
        render_messages(final_messages)
        render_analyst(final_messages)
        md = format_messages_as_markdown(
            final_messages,
            st.session_state.sim_topic,
            st.session_state.sim_num_rounds,
        )
        st.download_button(
            "📥 下载对话记录 (Markdown)",
            data=md,
            file_name="agent_crowd_simulation.md",
            mime="text/markdown",
        )

    elif st.session_state.finished_messages:
        st.success("✅ 模拟完成！")
        render_messages(st.session_state.finished_messages)
        render_analyst(st.session_state.finished_messages)
        md = format_messages_as_markdown(
            st.session_state.finished_messages,
            st.session_state.sim_topic,
            st.session_state.sim_num_rounds,
        )
        st.download_button(
            "📥 下载对话记录 (Markdown)",
            data=md,
            file_name="agent_crowd_simulation.md",
            mime="text/markdown",
        )

    else:
        st.info("👈 在左侧输入话题，点击「开始模拟」开始多智能体群体模拟")


main()
