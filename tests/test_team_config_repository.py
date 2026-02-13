"""Tests for lib_custom/team_config_repository.py."""

import json
from pathlib import Path

from lib_custom.personality_types import TeamConfig, TeamMember
from lib_custom.team_config_repository import TeamConfigRepository
import pytest


@pytest.fixture
def repo(tmp_path):
    """Create a repository pointing at a temp directory."""
    return TeamConfigRepository(config_path=str(tmp_path / "team_config.json"))


@pytest.fixture
def sample_config():
    return TeamConfig(
        boss_type_id="time_master",
        members=[
            TeamMember(id="a1", name="Alice", personality_type_id="strategic_charger"),
            TeamMember(id="b2", name="Bob", personality_type_id="zen_deadline"),
        ],
    )


class TestTeamConfigRepository:
    def test_load_nonexistent_returns_none(self, repo):
        assert repo.load_team_config() is None

    def test_save_and_load(self, repo, sample_config):
        repo.save_team_config(sample_config)
        loaded = repo.load_team_config()
        assert loaded is not None
        assert loaded.boss_type_id == "time_master"
        assert len(loaded.members) == 2

    def test_save_creates_json_file(self, repo, sample_config):
        repo.save_team_config(sample_config)
        assert Path(repo._path).exists()
        with open(repo._path) as f:
            data = json.load(f)
        assert data["boss_type_id"] == "time_master"

    def test_delete_config(self, repo, sample_config):
        repo.save_team_config(sample_config)
        assert Path(repo._path).exists()
        repo.delete_config()
        assert not Path(repo._path).exists()

    def test_delete_nonexistent_no_error(self, repo):
        repo.delete_config()  # should not raise

    def test_overwrite_config(self, repo, sample_config):
        repo.save_team_config(sample_config)
        new_config = TeamConfig(
            boss_type_id="time_chaos",
            members=[TeamMember(id="c3", name="Charlie", personality_type_id="blitz_executor")],
        )
        repo.save_team_config(new_config)
        loaded = repo.load_team_config()
        assert loaded is not None
        assert loaded.boss_type_id == "time_chaos"
        assert len(loaded.members) == 1

    def test_corrupt_file_raises(self, repo):
        Path(repo._path).parent.mkdir(parents=True, exist_ok=True)
        with open(repo._path, "w") as f:
            f.write("not valid json!!!")
        with pytest.raises(ValueError, match="Failed to load"):
            repo.load_team_config()

    def test_round_trip_preserves_data(self, repo, sample_config):
        repo.save_team_config(sample_config)
        loaded = repo.load_team_config()
        assert loaded is not None
        assert loaded.model_dump() == sample_config.model_dump()

    def test_empty_members(self, repo):
        config = TeamConfig(boss_type_id="time_master", members=[])
        repo.save_team_config(config)
        loaded = repo.load_team_config()
        assert loaded is not None
        assert loaded.members == []

    def test_chinese_names_preserved(self, repo):
        config = TeamConfig(
            boss_type_id="time_master",
            members=[TeamMember(id="x1", name="小明", personality_type_id="strategic_charger")],
        )
        repo.save_team_config(config)
        loaded = repo.load_team_config()
        assert loaded is not None
        assert loaded.members[0].name == "小明"
