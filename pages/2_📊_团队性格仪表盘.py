"""📊 团队性格仪表盘 — Streamlit page.

Five tabs:
1. 团队配置  – boss type + member management
2. 团队分析  – compatibility heatmap, balance radar, conflict alerts
3. 绩效预测  – project-type fitness, synergy gauge
4. 管理建议  – task assignment, communication, risk
5. 模拟对比  – what-if comparison
"""

from __future__ import annotations

import uuid

from lib_custom.engine.compatibility import (
    boss_compatibility_for_team,
    peer_compatibility_matrix,
)
from lib_custom.engine.conflicts import detect_conflicts
from lib_custom.engine.performance import PROJECT_TYPES, predict_performance
from lib_custom.engine.recommendations import generate_recommendations
from lib_custom.engine.team_balance import calculate_team_balance
from lib_custom.personality_types import (
    BOSS_TYPES,
    PERSONALITY_TYPES,
    TeamConfig,
    TeamMember,
)
from lib_custom.team_config_repository import TeamConfigRepository
import plotly.graph_objects as go  # type: ignore[import-untyped]
import streamlit as st


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="团队性格仪表盘", page_icon="📊", layout="wide")
st.title("📊 团队性格管理仪表盘")

_REPO = TeamConfigRepository()

# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = TeamConfig(boss_type_id="time_master", members=[])


def _get_config() -> TeamConfig:
    if "team_config" not in st.session_state:
        loaded = _REPO.load_team_config()
        st.session_state.team_config = loaded if loaded else _DEFAULT_CONFIG
    cfg: TeamConfig = st.session_state.team_config
    return cfg


def _set_config(cfg: TeamConfig) -> None:
    st.session_state.team_config = cfg
    _REPO.save_team_config(cfg)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🛠 团队配置",
    "🔍 团队分析",
    "📈 绩效预测",
    "💡 管理建议",
    "⚖️ 模拟对比",
])


# =========================================================================
# Tab 1 — Team Configuration
# =========================================================================
with tab1:
    config = _get_config()

    st.subheader("选择老板类型")
    boss_cols = st.columns(len(BOSS_TYPES))
    for idx, (bid, btype) in enumerate(BOSS_TYPES.items()):
        with boss_cols[idx]:
            selected = config.boss_type_id == bid
            border_color = "#4CAF50" if selected else "#ddd"
            st.markdown(
                f"""<div style="border:2px solid {border_color};border-radius:10px;
                padding:16px;text-align:center;">
                <h4>{'👔' if bid == 'time_master' else '🌀'} {btype.name_zh}</h4>
                <p style="font-size:0.85em;">{btype.description}</p>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button(
                "✅ 已选择" if selected else "选择",
                key=f"boss_{bid}",
                disabled=selected,
                use_container_width=True,
            ):
                _set_config(TeamConfig(
                    boss_type_id=bid,
                    members=config.members,
                ))
                st.rerun()

    st.divider()

    # --- Add member form ---
    st.subheader("添加员工")
    with st.form("add_member", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            member_name = st.text_input("姓名", max_chars=50)
        with c2:
            pt_options = {pid: f"{pt.icon} {pt.name_zh}" for pid, pt in PERSONALITY_TYPES.items()}
            selected_pt = st.selectbox("性格类型", options=list(pt_options.keys()), format_func=lambda x: pt_options[x])
        submitted = st.form_submit_button("➕ 添加", use_container_width=True)
        if submitted and member_name.strip():
            new_member = TeamMember(
                id=uuid.uuid4().hex[:8],
                name=member_name.strip(),
                personality_type_id=selected_pt,
                order=len(config.members),
            )
            _set_config(TeamConfig(
                boss_type_id=config.boss_type_id,
                members=[*config.members, new_member],
            ))
            st.rerun()

    # --- Member list ---
    st.subheader(f"当前团队 ({len(config.members)} 人)")
    if not config.members:
        st.info("尚未添加员工，请使用上方表单添加。")
    else:
        for member in config.members:
            pt = PERSONALITY_TYPES.get(member.personality_type_id)
            if pt is None:
                continue
            with st.container():
                mc1, mc2, mc3, mc4, mc5 = st.columns([2, 2, 2, 2, 1])
                mc1.markdown(f"**{pt.icon} {member.name}**")
                mc2.markdown(f"`{pt.name_zh}`")
                dim = pt.dimensions
                mc3.markdown(
                    f"紧迫:`{dim.urgency}` 行动:`{dim.action_pattern}` 时间:`{dim.time_orientation}`"
                )
                mc4.markdown(f"优势: {', '.join(pt.strengths[:2])}")
                if mc5.button("🗑️", key=f"del_{member.id}"):
                    _set_config(TeamConfig(
                        boss_type_id=config.boss_type_id,
                        members=[m for m in config.members if m.id != member.id],
                    ))
                    st.rerun()

    st.divider()
    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("💾 保存配置", use_container_width=True):
            _REPO.save_team_config(_get_config())
            st.success("已保存！")
    with sc2:
        if st.button("🔄 重置配置", use_container_width=True):
            _REPO.delete_config()
            st.session_state.pop("team_config", None)
            st.rerun()


# =========================================================================
# Tab 2 — Team Analysis
# =========================================================================
with tab2:
    config = _get_config()
    if len(config.members) < 2:
        st.warning("请在「团队配置」中添加至少 2 名员工后查看分析。")
    else:
        # --- Boss compatibility ---
        st.subheader("🤝 老板-员工兼容性")
        boss_scores = boss_compatibility_for_team(config)
        if boss_scores:
            fig_boss = go.Figure(go.Bar(
                x=[s.member_name for s in boss_scores],
                y=[s.score for s in boss_scores],
                text=[f"{s.score}" for s in boss_scores],
                textposition="outside",
                marker_color=[
                    "#4CAF50" if s.score >= 70 else "#FFC107" if s.score >= 50 else "#F44336"
                    for s in boss_scores
                ],
            ))
            boss_name = BOSS_TYPES[config.boss_type_id].name_zh
            fig_boss.update_layout(
                title=f"与「{boss_name}」的兼容性评分",
                yaxis_range=[0, 105],
                height=350,
            )
            st.plotly_chart(fig_boss, use_container_width=True)

        # --- Peer compatibility heatmap ---
        st.subheader("👥 同事间兼容性矩阵")
        peer_scores = peer_compatibility_matrix(config.members)
        if peer_scores:
            names = [m.name for m in config.members if m.personality_type_id in PERSONALITY_TYPES]
            n = len(names)
            matrix = [[0] * n for _ in range(n)]
            name_idx = {name: i for i, name in enumerate(names)}
            member_name_map = {m.id: m.name for m in config.members}
            for ps in peer_scores:
                a_name = member_name_map.get(ps.member_a_id, "")
                b_name = member_name_map.get(ps.member_b_id, "")
                if a_name in name_idx and b_name in name_idx:
                    i, j = name_idx[a_name], name_idx[b_name]
                    matrix[i][j] = ps.score
                    matrix[j][i] = ps.score
            for i in range(n):
                matrix[i][i] = 100

            fig_heat = go.Figure(go.Heatmap(
                z=matrix,
                x=names,
                y=names,
                colorscale="RdYlGn",
                zmin=0,
                zmax=100,
                text=matrix,
                texttemplate="%{text}",
            ))
            fig_heat.update_layout(title="同事兼容性热力图", height=400)
            st.plotly_chart(fig_heat, use_container_width=True)

        # --- Balance radar ---
        st.subheader("⚖️ 团队平衡分析")
        balance = calculate_team_balance(config.members)

        dim_labels = {
            "urgency": "紧迫感",
            "action_pattern": "行动模式",
            "time_orientation": "时间导向",
        }
        radar_categories: list[str] = []
        radar_values: list[float] = []
        for dist in balance.distributions:
            radar_categories.append(dim_labels.get(dist.dimension_name, dist.dimension_name))
            radar_values.append(dist.entropy * 100)

        fig_radar = go.Figure(go.Scatterpolar(
            r=[*radar_values, radar_values[0]],
            theta=[*radar_categories, radar_categories[0]],
            fill="toself",
            name="多样性",
        ))
        fig_radar.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 100]}},
            title=f"团队多样性得分: {balance.diversity_score:.0f}/100",
            height=400,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Dimension details
        for dist in balance.distributions:
            label = dim_labels.get(dist.dimension_name, dist.dimension_name)
            st.markdown(f"**{label}** (熵={dist.entropy:.2f}): {dict(dist.counts)}")

        # Missing values
        if balance.missing_values:
            st.warning("⚠️ 缺失维度值: " + str(balance.missing_values))

        # --- Conflict alerts ---
        st.subheader("⚠️ 冲突预警")
        conflicts = detect_conflicts(config)
        if not conflicts:
            st.success("未检测到显著冲突风险。")
        else:
            for alert in conflicts:
                icon = "🔴" if alert.severity == "high" else "🟡" if alert.severity == "medium" else "🟢"
                st.markdown(f"{icon} **{alert.title}**")
                st.caption(alert.description)

        # Warnings from balance
        if balance.warnings:
            for w in balance.warnings:
                st.markdown(f"🟡 {w}")


# =========================================================================
# Tab 3 — Performance Prediction
# =========================================================================
with tab3:
    config = _get_config()
    if not config.members:
        st.warning("请先在「团队配置」中添加员工。")
    else:
        st.subheader("选择项目类型")
        proj_cols = st.columns(len(PROJECT_TYPES))
        proj_id = st.session_state.get("selected_project", "urgent_launch")
        for idx, (pid, pinfo) in enumerate(PROJECT_TYPES.items()):
            with proj_cols[idx]:
                selected = proj_id == pid
                border = "#4CAF50" if selected else "#ddd"
                st.markdown(
                    f"""<div style="border:2px solid {border};border-radius:10px;
                    padding:14px;text-align:center;">
                    <h4>{pinfo['icon']} {pinfo['name_zh']}</h4>
                    <p style="font-size:0.85em;">{pinfo['description']}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if st.button(
                    "✅ 已选" if selected else "选择",
                    key=f"proj_{pid}",
                    disabled=selected,
                    use_container_width=True,
                ):
                    st.session_state.selected_project = pid
                    st.rerun()

        prediction = predict_performance(config, proj_id)

        # Overall gauge
        st.subheader("团队综合评分")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction.overall_score,
            title={"text": f"{prediction.project_name} — 综合适配度"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4CAF50" if prediction.overall_score >= 60 else "#FFC107"},
                "steps": [
                    {"range": [0, 40], "color": "#ffebee"},
                    {"range": [40, 70], "color": "#fff8e1"},
                    {"range": [70, 100], "color": "#e8f5e9"},
                ],
            },
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown(f"**{prediction.summary}**")

        # Individual fitness bar
        st.subheader("个人适配分")
        if prediction.individual_scores:
            fig_ind = go.Figure(go.Bar(
                x=[s.member_name for s in prediction.individual_scores],
                y=[s.score for s in prediction.individual_scores],
                text=[f"{s.score} ({s.detail})" for s in prediction.individual_scores],
                textposition="outside",
                marker_color=[
                    "#4CAF50" if s.score >= 70 else "#FFC107" if s.score >= 50 else "#F44336"
                    for s in prediction.individual_scores
                ],
            ))
            fig_ind.update_layout(yaxis_range=[0, 110], height=350)
            st.plotly_chart(fig_ind, use_container_width=True)

        # Synergy
        st.metric("团队协同分", prediction.team_synergy_score)


# =========================================================================
# Tab 4 — Management Recommendations
# =========================================================================
with tab4:
    config = _get_config()
    if not config.members:
        st.warning("请先在「团队配置」中添加员工。")
    else:
        proj_id = st.session_state.get("selected_project", "urgent_launch")
        recs = generate_recommendations(config, proj_id)

        categories = sorted({r.category for r in recs})
        for cat in categories:
            st.subheader(f"{'📋' if cat == '任务分配' else '💬' if cat == '沟通策略' else '⚠️'} {cat}")
            cat_recs = [r for r in recs if r.category == cat]
            for rec in cat_recs:
                priority_icon = "🔴" if rec.priority == 1 else "🟡" if rec.priority == 2 else "🟢"
                with st.expander(f"{priority_icon} {rec.title}", expanded=rec.priority == 1):
                    st.markdown(rec.description)
                    if rec.target_members:
                        member_map = {m.id: m.name for m in config.members}
                        names = [member_map.get(mid, mid) for mid in rec.target_members]
                        st.caption(f"涉及成员: {', '.join(names)}")


# =========================================================================
# Tab 5 — What-If Comparison
# =========================================================================
with tab5:
    config = _get_config()
    if not config.members:
        st.warning("请先在「团队配置」中添加员工。")
    else:
        st.subheader("模拟对比：切换老板 / 项目类型")
        cmp_c1, cmp_c2 = st.columns(2)
        with cmp_c1:
            alt_boss = st.selectbox(
                "对比老板类型",
                options=list(BOSS_TYPES.keys()),
                format_func=lambda x: BOSS_TYPES[x].name_zh,
                index=0 if config.boss_type_id != "time_master" else 1,
                key="alt_boss",
            )
        with cmp_c2:
            alt_proj = st.selectbox(
                "对比项目类型",
                options=list(PROJECT_TYPES.keys()),
                format_func=lambda x: PROJECT_TYPES[x]["name_zh"],
                key="alt_proj",
            )

        proj_id = st.session_state.get("selected_project", "urgent_launch")

        # Current
        current_pred = predict_performance(config, proj_id)

        # Alternative
        alt_config = TeamConfig(boss_type_id=alt_boss, members=config.members)
        alt_pred = predict_performance(alt_config, alt_proj)

        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown("### 当前方案")
            boss_name = BOSS_TYPES.get(config.boss_type_id)
            st.markdown(f"**老板:** {boss_name.name_zh if boss_name else config.boss_type_id}")
            pinfo = PROJECT_TYPES.get(proj_id, {})
            st.markdown(f"**项目:** {pinfo.get('name_zh', proj_id)}")
            st.metric("综合评分", current_pred.overall_score)
            st.metric("协同分", current_pred.team_synergy_score)

        with dc2:
            st.markdown("### 对比方案")
            alt_boss_obj = BOSS_TYPES.get(alt_boss)
            st.markdown(f"**老板:** {alt_boss_obj.name_zh if alt_boss_obj else alt_boss}")
            alt_pinfo = PROJECT_TYPES.get(alt_proj, {})
            st.markdown(f"**项目:** {alt_pinfo.get('name_zh', alt_proj)}")
            delta_overall = alt_pred.overall_score - current_pred.overall_score
            delta_synergy = alt_pred.team_synergy_score - current_pred.team_synergy_score
            st.metric("综合评分", alt_pred.overall_score, delta=delta_overall)
            st.metric("协同分", alt_pred.team_synergy_score, delta=delta_synergy)

        # Boss compatibility comparison
        st.subheader("老板兼容性对比")
        curr_boss_scores = boss_compatibility_for_team(config)
        alt_boss_scores = boss_compatibility_for_team(alt_config)

        if curr_boss_scores and alt_boss_scores:
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                name=f"当前 ({BOSS_TYPES[config.boss_type_id].name_zh})",
                x=[s.member_name for s in curr_boss_scores],
                y=[s.score for s in curr_boss_scores],
            ))
            fig_cmp.add_trace(go.Bar(
                name=f"对比 ({BOSS_TYPES[alt_boss].name_zh})",
                x=[s.member_name for s in alt_boss_scores],
                y=[s.score for s in alt_boss_scores],
            ))
            fig_cmp.update_layout(barmode="group", yaxis_range=[0, 105], height=350)
            st.plotly_chart(fig_cmp, use_container_width=True)
