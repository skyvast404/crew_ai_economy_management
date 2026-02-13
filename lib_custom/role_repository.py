"""Repository for role configuration persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import threading
from typing import Any

from lib_custom.role_models import RoleConfig, RolesDatabase, create_default_roles


logger = logging.getLogger(__name__)


class RoleRepository:
    """Thread-safe repository for role configurations."""

    def __init__(self, config_path: str = "roles_config.json"):
        """Initialize repository with config file path."""
        self.config_path = Path(config_path)
        self.backup_path = Path(f"{config_path}.backup")
        self._lock = threading.Lock()

    def load_roles(self) -> RolesDatabase:
        """Load roles from JSON file or create defaults."""
        with self._lock:
            if not self.config_path.exists():
                db = create_default_roles()
                self._save_without_lock(db)
                return db

            try:
                with open(self.config_path) as f:
                    data = json.load(f)
                db = RolesDatabase(**data)
                db.validate_database()
                return db
            except Exception as e:
                raise ValueError(f"Failed to load roles: {e}") from e

    def save_roles(self, db: RolesDatabase) -> None:
        """Save roles to JSON file with backup."""
        with self._lock:
            db.validate_database()
            self._create_backup()
            self._save_without_lock(db)

    def _save_without_lock(self, db: RolesDatabase) -> None:
        """Save without acquiring lock (internal use)."""
        temp_path = self.config_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(db.model_dump(), f, indent=2, ensure_ascii=False)
            temp_path.replace(self.config_path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise ValueError(f"Failed to save roles: {e}") from e

    def _create_backup(self) -> None:
        """Create backup of current config file."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    content = f.read()
                with open(self.backup_path, "w") as f:
                    f.write(content)
            except Exception:
                logger.warning("Failed to create backup", exc_info=True)

    def add_role(self, role: RoleConfig) -> RolesDatabase:
        """Add a new role to the database."""
        db = self.load_roles()

        # Check for duplicate role_id
        if any(r.role_id == role.role_id for r in db.roles):
            raise ValueError(f"Role with id '{role.role_id}' already exists")

        # Create new database with added role
        new_db = RolesDatabase(
            version=db.version,
            roles=[*db.roles, role]
        )

        self.save_roles(new_db)
        return new_db

    def reset_to_defaults(self) -> RolesDatabase:
        """Reset roles to default configuration."""
        db = create_default_roles()
        self.save_roles(db)
        return db

    def reorder_roles(self, role_orders: dict[str, int]) -> RolesDatabase:
        """Update the order of roles."""
        db = self.load_roles()

        new_roles = [
            role.model_copy(update={"order": role_orders[role.role_id]})
            if role.role_id in role_orders
            else role
            for role in db.roles
        ]

        new_db = RolesDatabase(version=db.version, roles=new_roles)
        self.save_roles(new_db)
        return new_db

    def update_role(self, role_id: str, updates: dict[str, Any]) -> RolesDatabase:
        """Update an existing role."""
        db = self.load_roles()

        # Find role index
        role_idx = next(
            (i for i, r in enumerate(db.roles) if r.role_id == role_id),
            None
        )

        if role_idx is None:
            raise ValueError(f"Role with id '{role_id}' not found")

        # Create updated role
        old_role = db.roles[role_idx]
        updated_data = old_role.model_dump()
        updated_data.update(updates)
        updated_role = RoleConfig(**updated_data)

        # Create new database with updated role
        new_roles = [*db.roles]
        new_roles[role_idx] = updated_role
        new_db = RolesDatabase(version=db.version, roles=new_roles)

        self.save_roles(new_db)
        return new_db

    def delete_role(self, role_id: str) -> RolesDatabase:
        """Delete a role from the database."""
        db = self.load_roles()

        # Find role
        role = next((r for r in db.roles if r.role_id == role_id), None)
        if not role:
            raise ValueError(f"Role with id '{role_id}' not found")

        # Check if role is default
        if role.is_default:
            raise ValueError("Cannot delete default roles")

        # Create new database without the role
        new_roles = [r for r in db.roles if r.role_id != role_id]
        new_db = RolesDatabase(version=db.version, roles=new_roles)

        self.save_roles(new_db)
        return new_db
