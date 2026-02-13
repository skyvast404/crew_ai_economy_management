"""Unit tests for role_repository.py"""

from pathlib import Path
import tempfile

from lib_custom.role_models import RoleConfig
from lib_custom.role_repository import RoleRepository
import pytest


class TestRoleRepository:
    """Test RoleRepository CRUD operations."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file path."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name
        # Delete the empty file so load_roles creates defaults
        Path(temp_path).unlink()
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)
        Path(f"{temp_path}.backup").unlink(missing_ok=True)

    def test_load_roles_creates_defaults(self, temp_config_file):
        """Test loading roles creates defaults if file doesn't exist."""
        repo = RoleRepository(temp_config_file)
        db = repo.load_roles()

        assert len(db.roles) == 4
        assert db.version == "1.0"
        assert Path(temp_config_file).exists()

    def test_save_and_load_roles(self, temp_config_file):
        """Test saving and loading roles."""
        repo = RoleRepository(temp_config_file)
        db = repo.load_roles()

        # Save should work
        repo.save_roles(db)

        # Load again should return same data
        db2 = repo.load_roles()
        assert len(db2.roles) == len(db.roles)
        assert db2.version == db.version

    def test_add_role(self, temp_config_file):
        """Test adding a new role."""
        repo = RoleRepository(temp_config_file)
        db = repo.load_roles()
        initial_count = len(db.roles)

        new_role = RoleConfig(
            role_id="test_role",
            role_name="Test Role",
            goal="Test goal for the role",
            backstory="Test backstory for the role",
            avatar="🎭",
            role_type="conversation",
        )

        new_db = repo.add_role(new_role)
        assert len(new_db.roles) == initial_count + 1

    def test_add_duplicate_role_fails(self, temp_config_file):
        """Test adding duplicate role fails."""
        repo = RoleRepository(temp_config_file)
        repo.load_roles()

        duplicate_role = RoleConfig(
            role_id="boss",  # Already exists
            role_name="Duplicate Boss",
            goal="Test goal for duplicate",
            backstory="Test backstory for duplicate",
            avatar="👔",
            role_type="conversation",
        )

        with pytest.raises(ValueError, match="already exists"):
            repo.add_role(duplicate_role)

    def test_update_role(self, temp_config_file):
        """Test updating an existing role."""
        repo = RoleRepository(temp_config_file)
        repo.load_roles()

        new_db = repo.update_role("boss", {"goal": "New goal for boss"})
        boss = next(r for r in new_db.roles if r.role_id == "boss")
        assert boss.goal == "New goal for boss"

    def test_update_nonexistent_role_fails(self, temp_config_file):
        """Test updating nonexistent role fails."""
        repo = RoleRepository(temp_config_file)
        repo.load_roles()

        with pytest.raises(ValueError, match="not found"):
            repo.update_role("nonexistent", {"goal": "New goal"})

    def test_delete_role(self, temp_config_file):
        """Test deleting a custom role."""
        repo = RoleRepository(temp_config_file)

        # Add a custom role first
        custom_role = RoleConfig(
            role_id="custom",
            role_name="Custom Role",
            goal="Custom goal",
            backstory="Custom backstory",
            avatar="🎪",
            role_type="conversation",
            is_default=False,
        )
        repo.add_role(custom_role)

        # Delete it
        new_db = repo.delete_role("custom")
        assert not any(r.role_id == "custom" for r in new_db.roles)

    def test_delete_default_role_fails(self, temp_config_file):
        """Test deleting default role fails."""
        repo = RoleRepository(temp_config_file)
        repo.load_roles()

        with pytest.raises(ValueError, match="Cannot delete default"):
            repo.delete_role("boss")

    def test_reset_to_defaults(self, temp_config_file):
        """Test resetting to default configuration."""
        repo = RoleRepository(temp_config_file)

        # Add custom role
        custom_role = RoleConfig(
            role_id="custom",
            role_name="Custom",
            goal="Custom goal",
            backstory="Custom backstory",
            avatar="🎪",
            role_type="conversation",
        )
        repo.add_role(custom_role)

        # Reset
        new_db = repo.reset_to_defaults()
        assert len(new_db.roles) == 4
        assert not any(r.role_id == "custom" for r in new_db.roles)

    def test_backup_created(self, temp_config_file):
        """Test backup file is created on save."""
        repo = RoleRepository(temp_config_file)
        repo.load_roles()

        # Modify and save
        repo.update_role("boss", {"goal": "Modified goal"})

        # Backup should exist
        assert Path(f"{temp_config_file}.backup").exists()
