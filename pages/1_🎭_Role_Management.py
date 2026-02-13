"""Role Management Page - CRUD interface for managing agent roles."""

from lib_custom.role_models import RoleConfig
from lib_custom.role_repository import RoleRepository
import streamlit as st


# Page config
st.set_page_config(
    page_title="角色管理",
    page_icon="🎭",
    layout="wide",
)

st.title("🎭 角色管理")
st.markdown("管理对话角色和分析师角色的配置")

# Initialize repository
repo = RoleRepository()

# Initialize session state
if "editing_role" not in st.session_state:
    st.session_state.editing_role = None
if "show_add_form" not in st.session_state:
    st.session_state.show_add_form = False


# Load roles
try:
    db = repo.load_roles()
except Exception as e:
    st.error(f"加载角色配置失败: {e}")
    st.stop()


# Action buttons
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("➕ 新增角色"):
        st.session_state.show_add_form = True
        st.rerun()
with col2:
    if st.button("🔄 重置为默认"):
        if st.session_state.get("confirm_reset"):
            try:
                repo.reset_to_defaults()
                st.success("已重置为默认配置")
                st.session_state.confirm_reset = False
                st.rerun()
            except Exception as e:
                st.error(f"重置失败: {e}")
        else:
            st.session_state.confirm_reset = True
            st.warning("再次点击确认重置")

st.divider()


# Display conversation roles
st.subheader("对话角色")
st.caption("顺序决定发言顺序")

conv_roles = db.get_conversation_roles()
for _idx, role in enumerate(conv_roles):
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.markdown(f"### {role.avatar} {role.role_name}")
            st.caption(f"**目标:** {role.goal[:50]}...")

        with col2:
            if st.button("✏️ 编辑", key=f"edit_{role.role_id}"):
                st.session_state.editing_role = role.role_id
                st.rerun()

        with col3:
            if not role.is_default:
                if st.button("🗑️ 删除", key=f"del_{role.role_id}"):
                    try:
                        repo.delete_role(role.role_id)
                        st.success(f"已删除角色: {role.role_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除失败: {e}")

        st.divider()


# Display analyst role
st.subheader("分析师角色")
analyst = db.get_analyst_role()
if analyst:
    with st.container():
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"### {analyst.avatar} {analyst.role_name}")
            st.caption(f"**目标:** {analyst.goal[:50]}...")

        with col2:
            if st.button("✏️ 编辑", key=f"edit_{analyst.role_id}"):
                st.session_state.editing_role = analyst.role_id
                st.rerun()

st.divider()


# Edit form modal
if st.session_state.editing_role:
    role_id = st.session_state.editing_role
    role = next((r for r in db.roles if r.role_id == role_id), None)

    if role:
        with st.form(key="edit_form"):
            st.subheader(f"编辑角色: {role.role_name}")

            # Basic info
            role_name = st.text_input("角色名称", value=role.role_name)
            avatar = st.text_input("头像 (1-2个字符)", value=role.avatar, max_chars=2)
            goal = st.text_area("目标", value=role.goal, height=100)
            backstory = st.text_area("背景故事", value=role.backstory, height=150)

            # Personality attributes (conversation roles only)
            if role.role_type == "conversation":
                st.divider()
                st.subheader("角色特征")
                personality = st.text_input(
                    "性格", value=role.personality or "",
                    help="如：果断、强势、目标导向",
                )
                communication_style = st.text_input(
                    "沟通风格", value=role.communication_style or "",
                    help="如：直接、简洁、命令式",
                )
                emotional_tendency = st.text_input(
                    "情绪倾向", value=role.emotional_tendency or "",
                    help="如：冷静但容易因进度问题焦虑",
                )
                values_field = st.text_input(
                    "价值观", value=role.values or "",
                    help="如：效率、结果、责任",
                )

            st.divider()
            st.subheader("提示词模板")

            # Prompt templates
            if role.role_type == "conversation":
                round_1_prompt = st.text_area(
                    "第一轮提示词",
                    value=role.round_1_prompt or "",
                    height=200,
                    help="可用变量: {role_name}, {goal}, {backstory}, {topic}"
                )
                followup_prompt = st.text_area(
                    "后续轮提示词",
                    value=role.followup_prompt or "",
                    height=200,
                    help="可用变量: {role_name}, {goal}, {backstory}, {topic}, {round}, {context}"
                )
            else:
                analyst_prompt = st.text_area(
                    "分析师提示词",
                    value=role.analyst_prompt or "",
                    height=300,
                    help="可用变量: {topic}, {num_rounds}, {full_conversation}"
                )

            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("💾 保存"):
                    try:
                        updates = {
                            "role_name": role_name,
                            "avatar": avatar,
                            "goal": goal,
                            "backstory": backstory,
                        }

                        if role.role_type == "conversation":
                            updates["round_1_prompt"] = round_1_prompt or None
                            updates["followup_prompt"] = followup_prompt or None
                            updates["personality"] = personality or None
                            updates["communication_style"] = communication_style or None
                            updates["emotional_tendency"] = emotional_tendency or None
                            updates["values"] = values_field or None
                        else:
                            updates["analyst_prompt"] = analyst_prompt or None

                        repo.update_role(role_id, updates)
                        st.success("保存成功!")
                        st.session_state.editing_role = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"保存失败: {e}")

            with col2:
                if st.form_submit_button("❌ 取消"):
                    st.session_state.editing_role = None
                    st.rerun()


# Add form modal
if st.session_state.show_add_form:
    with st.form(key="add_form"):
        st.subheader("新增角色")

        # Basic info
        role_id = st.text_input("角色ID (英文字母和下划线)", placeholder="my_role")
        role_name = st.text_input("角色名称", placeholder="我的角色")
        avatar = st.text_input("头像 (1-2个字符)", placeholder="🎭", max_chars=2)
        role_type = st.selectbox("角色类型", ["conversation", "analyst"])
        goal = st.text_area("目标", height=100)
        backstory = st.text_area("背景故事", height=150)

        # Personality attributes (conversation roles only)
        if role_type == "conversation":
            st.divider()
            st.subheader("角色特征 (可选)")
            personality = st.text_input(
                "性格", placeholder="如：果断、强势、目标导向", key="add_personality",
            )
            communication_style = st.text_input(
                "沟通风格", placeholder="如：直接、简洁、命令式", key="add_comm_style",
            )
            emotional_tendency = st.text_input(
                "情绪倾向", placeholder="如：冷静但容易因进度问题焦虑", key="add_emotion",
            )
            values_field = st.text_input(
                "价值观", placeholder="如：效率、结果、责任", key="add_values",
            )

        st.divider()
        st.subheader("提示词模板 (可选)")

        if role_type == "conversation":
            round_1_prompt = st.text_area(
                "第一轮提示词",
                height=200,
                help="可用变量: {role_name}, {goal}, {backstory}, {topic}"
            )
            followup_prompt = st.text_area(
                "后续轮提示词",
                height=200,
                help="可用变量: {role_name}, {goal}, {backstory}, {topic}, {round}, {context}"
            )
        else:
            analyst_prompt = st.text_area(
                "分析师提示词",
                height=300,
                help="可用变量: {topic}, {num_rounds}, {full_conversation}"
            )

        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("➕ 添加"):
                try:
                    new_role = RoleConfig(
                        role_id=role_id,
                        role_name=role_name,
                        avatar=avatar,
                        role_type=role_type,
                        goal=goal,
                        backstory=backstory,
                        is_default=False,
                        order=len(db.roles),
                    )

                    if role_type == "conversation":
                        new_role.round_1_prompt = round_1_prompt or None
                        new_role.followup_prompt = followup_prompt or None
                        new_role.personality = personality or None
                        new_role.communication_style = communication_style or None
                        new_role.emotional_tendency = emotional_tendency or None
                        new_role.values = values_field or None
                    else:
                        new_role.analyst_prompt = analyst_prompt or None

                    repo.add_role(new_role)
                    st.success("添加成功!")
                    st.session_state.show_add_form = False
                    st.rerun()
                except Exception as e:
                    st.error(f"添加失败: {e}")

        with col2:
            if st.form_submit_button("❌ 取消"):
                st.session_state.show_add_form = False
                st.rerun()
