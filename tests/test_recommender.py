from pathlib import Path
import tempfile
import unittest

from recommender import find_reference_images, rank_character_scores


class FindReferenceImagesTests(unittest.TestCase):
    def test_groups_supported_files_by_character_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_character = root / "Mina"
            first_character.mkdir()
            (first_character / "01.JPG").touch()
            (first_character / "notes.txt").touch()
            (root / "loose.png").touch()
            empty_character = root / "Empty"
            empty_character.mkdir()

            found = find_reference_images(root)

            self.assertEqual(list(found), ["Mina"])
            self.assertEqual([path.name for path in found["Mina"]], ["01.JPG"])

    def test_missing_directory_has_no_references(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing"
            self.assertEqual(find_reference_images(missing), {})


class RankCharacterScoresTests(unittest.TestCase):
    # 수정: 2026-10-27 16:25 KST - 상위 참고 사진 평균 점수와 단일 고득점 완화 동작을 검사합니다.
    def test_ranks_by_average_of_best_reference_images(self) -> None:
        references = {
            "third": [[0.0, 1.0]],
            "first": [[0.8, 0.6], [0.8, -0.6]],
            "second": [[0.6, 0.8]],
            "fourth": [[-1.0, 0.0]],
        }

        ranked = rank_character_scores(references, [1.0, 0.0], limit=3)

        self.assertEqual([name for name, _ in ranked], ["first", "second", "third"])
        self.assertEqual(ranked[0][1], 0.8)

    def test_single_high_score_does_not_dominate_many_references(self) -> None:
        references = {
            "one_lucky_match": [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            "consistently_similar": [[0.8, 0.6], [0.8, -0.6], [0.8, 0.6]],
        }

        ranked = rank_character_scores(references, [1.0, 0.0])

        self.assertEqual(ranked[0][0], "consistently_similar")
        self.assertAlmostEqual(ranked[0][1], 0.8)

    def test_rejects_mismatched_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "same dimensions"):
            rank_character_scores({"Mina": [[1.0, 0.0]]}, [1.0])

    def test_rejects_invalid_result_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            rank_character_scores({"Mina": [[1.0]]}, [1.0], limit=0)

    def test_rejects_zero_vectors(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero magnitude"):
            rank_character_scores({"Mina": [[0.0, 0.0]]}, [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
