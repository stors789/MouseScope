from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from services.project_store import ProjectStore


class ProjectStoreWorkflowTest(unittest.TestCase):
    def test_project_video_assignment_and_cleanup_workflow(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data" / "projects"
            store = ProjectStore(root, now_fn=lambda: 1000.0)
            project = store.create_project("workflow")

            group = store.add_group(project["project_id"], "control")
            subject = store.add_subject(
                project["project_id"],
                {"animal_code": "M001", "group_id": group["group_id"]},
            )
            self.assertEqual(subject["animal_code"], "M001")

            generated = store.add_subjects_to_group(project["project_id"], group["group_id"], 3)
            self.assertEqual([item["animal_code"] for item in generated], ["control1", "control2", "control3"])
            more_generated = store.add_subjects_to_group(project["project_id"], group["group_id"], 2)
            self.assertEqual([item["animal_code"] for item in more_generated], ["control4", "control5"])
            renamed = store.update_subject(
                project["project_id"],
                generated[0]["subject_id"],
                {"animal_code": "C-A"},
            )
            self.assertEqual(renamed["animal_code"], "C-A")

            original = Path(temp_dir) / "original.mp4"
            original.write_bytes(b"linked-video")
            linked_video = store.add_video(
                project["project_id"],
                str(original),
                original_name="original.mp4",
                storage_mode="linked",
            )
            assignment = store.create_assignment(
                project["project_id"],
                {"video_id": linked_video["video_id"], "experiment_type": "tst"},
            )
            self.assertEqual(len(store.list_assignments(project["project_id"])), 1)

            self.assertTrue(store.delete_assignment(project["project_id"], assignment["assignment_id"]))
            self.assertEqual(store.list_assignments(project["project_id"]), [])
            self.assertTrue(original.exists())

            removable_video = store.add_video(
                project["project_id"],
                str(original),
                original_name="remove-me.mp4",
                storage_mode="linked",
            )
            removable_assignment = store.create_assignment(
                project["project_id"],
                {"video_id": removable_video["video_id"], "experiment_type": "tst"},
            )
            deleted = store.delete_video(project["project_id"], removable_video["video_id"])
            self.assertEqual(deleted["assignment_ids"], [removable_assignment["assignment_id"]])
            self.assertIsNone(store.get_video(project["project_id"], removable_video["video_id"]))
            self.assertEqual(store.list_assignments(project["project_id"]), [])
            self.assertTrue(original.exists())

            staged_dir = root.parent / "uploads" / "_staged" / "upload1"
            staged_dir.mkdir(parents=True)
            delete_copy_source = staged_dir / "delete-copy.mp4"
            delete_copy_source.write_bytes(b"delete-copy-video")
            delete_copy_video = store.add_video(
                project["project_id"],
                str(delete_copy_source),
                original_name="delete-copy.mp4",
                storage_mode="copied",
                notes="upload_cache:auto_cleanup",
            )
            delete_copy_path = Path(delete_copy_video["project_path"])
            self.assertTrue(delete_copy_path.exists())
            store.delete_video(project["project_id"], delete_copy_video["video_id"])
            self.assertFalse(delete_copy_path.exists())
            self.assertFalse(delete_copy_source.exists())

            staged_dir.mkdir(parents=True, exist_ok=True)
            uploaded = staged_dir / "uploaded.mp4"
            uploaded.write_bytes(b"uploaded-video")
            copied_video = store.add_video(
                project["project_id"],
                str(uploaded),
                original_name="uploaded.mp4",
                storage_mode="copied",
                notes="upload_cache:auto_cleanup",
            )
            copied_assignment = store.create_assignment(
                project["project_id"],
                {"video_id": copied_video["video_id"], "experiment_type": "open_field"},
            )
            project_copy = Path(copied_video["project_path"])
            self.assertTrue(project_copy.exists())

            uploaded.unlink()
            cleaned = store.cleanup_completed_upload_caches(
                project["project_id"],
                [copied_assignment["assignment_id"]],
            )
            self.assertEqual(len(cleaned), 1)
            self.assertFalse(project_copy.exists())
            refreshed = store.get_video(project["project_id"], copied_video["video_id"])
            self.assertEqual(refreshed["storage_mode"], "cleaned")
            self.assertEqual(refreshed["path_exists"], 0)


if __name__ == "__main__":
    unittest.main()
