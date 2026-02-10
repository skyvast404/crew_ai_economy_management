"""Unit tests for role_models.py"""

import pytest
from pydantic import ValidationError
from lib_custom.role_models import (
    RoleConfig,
    RolesDatabase,
    create_default_roles,
    DEFAULT_ROUND_1_PROMPT,
    DEFAULT_FOLLOWUP_PROMPT,
    DEFAULT_ANALYST_PROMPT,
)


class TestRoleConfig:
    """Test RoleConfig validation."""

    def test_valid_role_config(self):
        """Test creating a valid role config."""
        role = RoleConfig(
            role_id="test_role",
            role_name="Test Role",
            goal="Test goal for the role",
            backstory="Test backstory for the role",
            avatar="🎭",
            role_type="conversation",
            order=0,
        )
        assert role.role_id == "test_role"
        assert role.role_name == "Test Role"
        assert role.role_type == "conversation"

    def test_role_id_validation_alphanumeric(self):
        """Test role_id must be alphanumeric with underscores."""
        with pytest.raises(ValidationError):
            RoleConfig(
                role_id="test-role",  # Hyphens not allowed
                role_name="Test Role",
                goal="Test goal for the role",
                backstory="Test backstory for the role",
                avatar="🎭",
                role_type="conversation",
            )

    def test_avatar_length_validation(self):
        """Test avatar must be 1-2 characters."""
        with pytest.raises(ValidationError):
            RoleConfig(
                role_id="test_role",
                role_name="Test Role",
                goal="Test goal for the role",
                backstory="Test backstory for the role",
                avatar="🎭🎭🎭",  # Too long
                role_type="conversation",
            )

    def test_field_length_constraints(self):
        """Test field length constraints."""
        with pytest.raises(ValidationError):
            RoleConfig(
                role_id="test_role",
                role_name="Test Role",
                goal="Short",  # Too short (min 10 chars)
                backstory="Test backstory for the role",
                avatar="🎭",
                role_type="conversation",
            )

    def test_prompt_templates_optional(self):
        """Test prompt templates are optional."""
        role = RoleConfig(
            role_id="test_role",
            role_name="Test Role",
            goal="Test goal for the role",
            backstory="Test backstory for the role",
            avatar="🎭",
            role_type="conversation",
        )
        assert role.round_1_prompt is None
        assert role.followup_prompt is None
        assert role.analyst_prompt is None


class TestRolesDatabase:
    """Test RolesDatabase validation."""

    def test_get_conversation_roles(self):
        """Test getting conversation roles sorted by order."""
        db = create_default_roles()
        conv_roles = db.get_conversation_roles()
        assert len(conv_roles) == 3
        assert conv_roles[0].role_id == "boss"
        assert conv_roles[1].role_id == "senior"
        assert conv_roles[2].role_id == "newbie"

    def test_get_analyst_role(self):
        """Test getting analyst role."""
        db = create_default_roles()
        analyst = db.get_analyst_role()
        assert analyst is not None
        assert analyst.role_id == "analyst"
        assert analyst.role_type == "analyst"

    def test_validate_unique_role_ids(self):
        """Test validation fails with duplicate role_ids."""
        role1 = RoleConfig(
            role_id="duplicate",
            role_name="Role 1",
            goal="Test goal for role 1",
            backstory="Test backstory for role 1",
            avatar="🎭",
            role_type="conversation",
        )
        role2 = RoleConfig(
            role_id="duplicate",
            role_name="Role 2",
            goal="Test goal for role 2",
            backstory="Test backstory for role 2",
            avatar="🎪",
            role_type="conversation",
        )
        db = RolesDatabase(roles=[role1, role2])
        with pytest.raises(ValueError, match="Duplicate role_ids"):
            db.validate_database()

    def test_validate_minimum_conversation_roles(self):
        """Test validation fails with less than 2 conversation roles."""
        role1 = RoleConfig(
            role_id="conv1",
            role_name="Conv 1",
            goal="Test goal for conv 1",
            backstory="Test backstory for conv 1",
            avatar="🎭",
            role_type="conversation",
        )
        analyst = RoleConfig(
            role_id="analyst",
            role_name="Analyst",
            goal="Test goal for analyst",
            backstory="Test backstory for analyst",
            avatar="📊",
            role_type="analyst",
        )
        db = RolesDatabase(roles=[role1, analyst])
        with pytest.raises(ValueError, match="At least 2 conversation roles"):
            db.validate_database()

    def test_validate_exactly_one_analyst(self):
        """Test validation fails without exactly one analyst."""
        role1 = RoleConfig(
            role_id="conv1",
            role_name="Conv 1",
            goal="Test goal for conv 1",
            backstory="Test backstory for conv 1",
            avatar="🎭",
            role_type="conversation",
        )
        role2 = RoleConfig(
            role_id="conv2",
            role_name="Conv 2",
            goal="Test goal for conv 2",
            backstory="Test backstory for conv 2",
            avatar="🎪",
            role_type="conversation",
        )
        db = RolesDatabase(roles=[role1, role2])
        with pytest.raises(ValueError, match="Exactly 1 analyst role"):
            db.validate_database()


class TestDefaultRoles:
    """Test default roles factory."""

    def test_create_default_roles(self):
        """Test creating default roles."""
        db = create_default_roles()
        assert len(db.roles) == 4
        assert db.version == "1.0"

    def test_default_roles_valid(self):
        """Test default roles pass validation."""
        db = create_default_roles()
        db.validate_database()  # Should not raise

    def test_default_roles_have_prompts(self):
        """Test default roles have prompt templates."""
        db = create_default_roles()
        conv_roles = db.get_conversation_roles()
        for role in conv_roles:
            assert role.round_1_prompt is not None
            assert role.followup_prompt is not None

        analyst = db.get_analyst_role()
        assert analyst.analyst_prompt is not None
