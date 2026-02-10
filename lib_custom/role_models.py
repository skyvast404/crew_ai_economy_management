"""Pydantic models for role configuration and validation."""

from typing import Literal
from pydantic import BaseModel, Field, field_validator


class RoleConfig(BaseModel):
    """Configuration for a single role (agent)."""

    role_id: str = Field(..., min_length=1, max_length=50)
    role_name: str = Field(..., min_length=1, max_length=100)
    goal: str = Field(..., min_length=10, max_length=500)
    backstory: str = Field(..., min_length=10, max_length=1000)
    avatar: str = Field(..., min_length=1, max_length=2)
    role_type: Literal["conversation", "analyst"]
    is_default: bool = False
    order: int = Field(default=0, ge=0)

    # Prompt templates
    round_1_prompt: str | None = None
    followup_prompt: str | None = None
    analyst_prompt: str | None = None

    @field_validator("role_id")
    @classmethod
    def validate_role_id(cls, v: str) -> str:
        """Validate role_id is alphanumeric with underscores."""
        if not v.replace("_", "").isalnum():
            raise ValueError("role_id must be alphanumeric with underscores")
        return v

    @field_validator("avatar")
    @classmethod
    def validate_avatar(cls, v: str) -> str:
        """Validate avatar is 1-2 characters."""
        if len(v) < 1 or len(v) > 2:
            raise ValueError("avatar must be 1-2 characters")
        return v


class RolesDatabase(BaseModel):
    """Database of all roles with validation."""

    version: str = "1.0"
    roles: list[RoleConfig]

    def get_conversation_roles(self) -> list[RoleConfig]:
        """Get all conversation roles sorted by order."""
        return sorted(
            [r for r in self.roles if r.role_type == "conversation"],
            key=lambda r: r.order
        )

    def get_analyst_role(self) -> RoleConfig | None:
        """Get the analyst role."""
        analysts = [r for r in self.roles if r.role_type == "analyst"]
        return analysts[0] if analysts else None

    def validate_database(self) -> None:
        """Validate database constraints."""
        # Check unique role_ids
        role_ids = [r.role_id for r in self.roles]
        if len(role_ids) != len(set(role_ids)):
            raise ValueError("Duplicate role_ids found")

        # Check minimum conversation roles
        conv_roles = self.get_conversation_roles()
        if len(conv_roles) < 2:
            raise ValueError("At least 2 conversation roles required")

        # Check exactly one analyst
        analysts = [r for r in self.roles if r.role_type == "analyst"]
        if len(analysts) != 1:
            raise ValueError("Exactly 1 analyst role required")


# Default prompt templates
DEFAULT_ROUND_1_PROMPT = """你是{role_name}。

你的目标: {goal}

你的背景: {backstory}

当前会议主题: {topic}

请根据你的角色定位，发表你对这个主题的初步看法。注意保持你的角色特征和说话风格。"""

DEFAULT_FOLLOWUP_PROMPT = """你是{role_name}。

会议主题: {topic}

之前的讨论:
{context}

请根据之前的讨论，继续发表你的看法。注意:
1. 回应其他人的观点
2. 保持你的角色特征
3. 推进讨论或维护你的立场"""

DEFAULT_ANALYST_PROMPT = """你是组织行为学研究者。请分析以下会议讨论:

会议主题: {topic}

完整对话记录:
{full_conversation}

请从以下4个模块进行学术分析:

模块1: 硬指标 (绩效影响)
- 任务绩效影响
- 组织绩效影响

模块2: 软指标 (行为变量)
- OCB (组织公民行为)
- 创新行为
- UPB (非伦理亲组织行为)
- 离职倾向信号

模块3: 变量关系建模
- 自变量 (IV)
- 因变量 (DV)
- 中介变量
- 调节变量
- 研究假设

模块4: 测量与方法论
- 推荐量表
- 适用理论
- 研究模型路径

请提供结构化的学术分析报告。"""


def create_default_roles() -> RolesDatabase:
    """Create default roles configuration."""
    roles = [
        RoleConfig(
            role_id="boss",
            role_name="👔 老板 (Boss)",
            goal="推动项目按时交付，确保团队高效运作",
            backstory="你是公司的创始人，对项目成功负有最终责任。你关注结果和效率。",
            avatar="👔",
            role_type="conversation",
            is_default=True,
            order=0,
            round_1_prompt=DEFAULT_ROUND_1_PROMPT,
            followup_prompt=DEFAULT_FOLLOWUP_PROMPT,
        ),
        RoleConfig(
            role_id="senior",
            role_name="🦊 资深员工 (Senior)",
            goal="平衡工作质量与个人生活，维护团队稳定",
            backstory="你在公司工作多年，经验丰富但也看透了很多职场规则。你懂得保护自己。",
            avatar="🦊",
            role_type="conversation",
            is_default=True,
            order=1,
            round_1_prompt=DEFAULT_ROUND_1_PROMPT,
            followup_prompt=DEFAULT_FOLLOWUP_PROMPT,
        ),
        RoleConfig(
            role_id="newbie",
            role_name="🐣 新人 (Newbie)",
            goal="快速学习和成长，获得认可",
            backstory="你刚加入公司不久，充满热情但缺乏经验。你渴望证明自己的价值。",
            avatar="🐣",
            role_type="conversation",
            is_default=True,
            order=2,
            round_1_prompt=DEFAULT_ROUND_1_PROMPT,
            followup_prompt=DEFAULT_FOLLOWUP_PROMPT,
        ),
        RoleConfig(
            role_id="analyst",
            role_name="📊 分析师 (Analyst)",
            goal="从组织行为学角度分析会议讨论",
            backstory="你是组织行为学研究者，擅长分析职场互动和组织动态。",
            avatar="📊",
            role_type="analyst",
            is_default=True,
            order=999,
            analyst_prompt=DEFAULT_ANALYST_PROMPT,
        ),
    ]

    return RolesDatabase(roles=roles)
