"""Leadership style definitions and role transformation utilities."""

from pydantic import BaseModel, Field

from lib_custom.role_models import RolesDatabase


class LeadershipStyle(BaseModel):
    """A leadership style that overrides boss role attributes."""

    style_id: str = Field(..., min_length=1, max_length=50)
    style_name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=10)
    boss_goal: str = Field(..., min_length=10)
    boss_backstory: str = Field(..., min_length=10)
    boss_personality: str = Field(..., min_length=2)
    boss_communication_style: str = Field(..., min_length=2)
    boss_emotional_tendency: str = Field(..., min_length=2)
    boss_values: str = Field(..., min_length=2)


LEADERSHIP_STYLES: dict[str, LeadershipStyle] = {
    "transformational": LeadershipStyle(
        style_id="transformational",
        style_name="变革型领导",
        description="通过愿景激励和个性化关怀来激发团队潜能，鼓励创新和超越自我",
        boss_goal="用愿景激励团队，激发每个人的潜能，推动组织变革与创新",
        boss_backstory="你是一位富有远见的领导者，相信每个人都有超越自我的潜力。你善于描绘愿景，用个人魅力感染团队。",
        boss_personality="富有魅力、理想主义、善于激励",
        boss_communication_style="鼓舞人心、善于讲故事、关注个人成长",
        boss_emotional_tendency="热情洋溢、充满感染力、对团队充满信心",
        boss_values="愿景、创新、个人成长、团队潜能",
    ),
    "transactional": LeadershipStyle(
        style_id="transactional",
        style_name="交易型领导",
        description="通过明确的奖惩机制和绩效标准来管理团队，强调规则和目标达成",
        boss_goal="建立清晰的绩效标准和奖惩机制，确保团队按规则高效运作",
        boss_backstory="你是一位注重制度和流程的管理者，相信明确的规则和公平的奖惩是团队运作的基础。",
        boss_personality="理性、严谨、注重规则",
        boss_communication_style="直接、数据驱动、强调KPI和目标",
        boss_emotional_tendency="冷静克制、就事论事、不感情用事",
        boss_values="公平、效率、制度、绩效",
    ),
    "servant": LeadershipStyle(
        style_id="servant",
        style_name="服务型领导",
        description="以服务团队为核心，优先满足成员需求，通过赋能和支持来提升团队效能",
        boss_goal="服务团队成员，移除障碍，创造让每个人都能发挥最佳状态的环境",
        boss_backstory="你相信领导的本质是服务。你把团队成员的成长和幸福放在首位，通过倾听和支持来带领团队。",
        boss_personality="谦逊、同理心强、乐于助人",
        boss_communication_style="倾听为主、温和询问、鼓励表达",
        boss_emotional_tendency="温暖关怀、善于共情、耐心包容",
        boss_values="服务、赋能、团队幸福、共同成长",
    ),
    "authoritative": LeadershipStyle(
        style_id="authoritative",
        style_name="权威型领导",
        description="依靠职位权力和严格控制来管理团队，强调服从和执行力",
        boss_goal="确保团队绝对服从指令，维护组织纪律和执行效率",
        boss_backstory="你是一位强势的管理者，相信明确的等级制度和严格的纪律是组织成功的关键。你的话就是命令。",
        boss_personality="强势、控制欲强、不容置疑",
        boss_communication_style="命令式、单向沟通、不接受反驳",
        boss_emotional_tendency="威严、易怒、对失误零容忍",
        boss_values="服从、纪律、权威、执行力",
    ),
}


def apply_style_to_roles(db: RolesDatabase, style: LeadershipStyle) -> RolesDatabase:
    """Immutably apply a leadership style to the boss role in a RolesDatabase.

    Returns a new RolesDatabase with the boss role's attributes overridden.
    """
    new_roles = []
    for role in db.roles:
        if role.role_id == "boss":
            new_roles.append(role.model_copy(update={
                "goal": style.boss_goal,
                "backstory": style.boss_backstory,
                "personality": style.boss_personality,
                "communication_style": style.boss_communication_style,
                "emotional_tendency": style.boss_emotional_tendency,
                "values": style.boss_values,
            }))
        else:
            new_roles.append(role)

    return RolesDatabase(version=db.version, roles=new_roles)


# ---------------------------------------------------------------------------
# Boss-type → leadership-style mapping
# ---------------------------------------------------------------------------
BOSS_TYPE_TO_LEADERSHIP_STYLES: dict[str, list[str]] = {
    "time_master": ["transformational", "servant"],
    "time_chaos": ["transactional", "authoritative"],
}


def get_leadership_styles_for_boss(boss_type_id: str) -> list[LeadershipStyle]:
    """Return the leadership styles associated with a boss type."""
    style_ids = BOSS_TYPE_TO_LEADERSHIP_STYLES.get(boss_type_id, [])
    return [LEADERSHIP_STYLES[sid] for sid in style_ids if sid in LEADERSHIP_STYLES]


DEFAULT_COMPARISON_ANALYST_PROMPT = """你是组织行为学研究者。请对比分析同一话题在不同领导风格下的团队互动差异。

会议主题: {topic}

以下是各领导风格下的完整对话记录:

{style_conversations}

请从以下维度进行跨风格对比分析:

1. 团队氛围差异
   - 各风格下团队的心理安全感水平
   - 成员发言积极性和开放程度

2. 决策质量对比
   - 各风格下讨论的深度和广度
   - 创新观点的产生数量和质量

3. 权力动态分析
   - 领导者与成员之间的互动模式
   - 成员之间的互动模式变化

4. 行为变量对比
   - OCB (组织公民行为) 表现差异
   - 创新行为差异
   - 离职倾向信号差异

5. 综合评估
   - 各领导风格的优劣势总结
   - 适用场景建议
   - 对组织管理的启示

请提供结构化的跨风格对比分析报告。"""
