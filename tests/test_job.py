from src.core.job import RunJob


class TestGetAnsweredTime:
    def test_empty_results(self, job: RunJob):
        assert job.get_answered_time() == 0.0

    def test_sums_took_across_models(self, job: RunJob):
        job.add_result("gpt-4", "q1", True, "a", 1.5, 50)
        job.add_result("gpt-4", "q2", False, "b", 2.25, 60)
        job.add_result("llama", "q1", True, "c", 3.0, 70)
        assert job.get_answered_time() == 6.75


class TestAddResult:
    def test_first_result_creates_model_entry(self, job: RunJob):
        job.add_result(
            model="gpt-4",
            question_uuid="q1",
            succeeded=True,
            response_text="hello",
            took=1.5,
            tokens=100,
        )
        assert len(job.results) == 1
        entry = job.results[0]
        assert entry["llm"] == "gpt-4"
        assert entry["score"] == "1/1"
        assert len(entry["responses"]) == 1
        resp = entry["responses"][0]
        assert resp["uuid"] == "q1"
        assert resp["response"] == "hello"
        assert resp["took"] == 1.5
        assert resp["consumedTokens"] == 100
        assert resp["completionTokens"] == 100
        assert resp["promptTokens"] == 0
        assert resp["ttft"] == 0.0
        assert resp["genDuration"] == 0.0
        assert resp["tokensEstimated"] is False
        assert resp["succeeded"] is True

    def test_timing_fields_are_stored(self, job: RunJob):
        job.add_result(
            model="gpt-4",
            question_uuid="q1",
            succeeded=True,
            response_text="hello",
            took=2.0,
            tokens=30,
            ttft=0.4,
            gen_duration=1.5,
            prompt_tokens=12,
            tokens_estimated=True,
        )
        resp = job.results[0]["responses"][0]
        assert resp["ttft"] == 0.4
        assert resp["genDuration"] == 1.5
        assert resp["promptTokens"] == 12
        assert resp["tokensEstimated"] is True

    def test_second_result_updates_score(self, job: RunJob):
        job.add_result("gpt-4", "q1", True, "a", 1.0, 50)
        job.add_result("gpt-4", "q2", False, "b", 2.0, 60)
        entry = job.results[0]
        assert entry["score"] == "1/2"
        assert len(entry["responses"]) == 2

    def test_different_models_are_separate(self, job: RunJob):
        job.add_result("gpt-4", "q1", True, "a", 1.0, 50)
        job.add_result("llama", "q1", False, "b", 2.0, 60)
        assert len(job.results) == 2
        gpt = next(e for e in job.results if e["llm"] == "gpt-4")
        llama = next(e for e in job.results if e["llm"] == "llama")
        assert gpt["score"] == "1/1"
        assert llama["score"] == "0/1"


class TestGetOverallStats:
    def test_empty_results(self, job: RunJob):
        assert job.get_overall_stats() == (0, 0)

    def test_counts_across_models(self, job: RunJob):
        job.add_result("gpt-4", "q1", True, "a", 1.0, 50)
        job.add_result("gpt-4", "q2", False, "b", 2.0, 60)
        job.add_result("llama", "q1", True, "c", 1.5, 70)
        assert job.get_overall_stats() == (2, 3)


class TestGetConsumedTokens:
    def test_empty_results(self, job: RunJob):
        assert job.get_consumed_tokens() == 0

    def test_sums_tokens_across_models(self, job: RunJob):
        job.add_result("gpt-4", "q1", True, "a", 1.5, 50)
        job.add_result("gpt-4", "q2", False, "b", 2.25, 60)
        job.add_result("llama", "q1", True, "c", 3.0, 70)
        assert job.get_consumed_tokens() == 180


class TestGetOverallTokenRate:
    def test_zero_when_nothing_answered(self, job: RunJob):
        assert job.get_overall_token_rate() == 0.0

    def test_zero_without_measurable_generation(self, job: RunJob):
        job.add_result("gpt-4", "q1", True, "a", 2.0, 100)
        assert job.get_overall_token_rate() == 0.0

    def test_generation_rate_excludes_ttft_and_verification(self, job: RunJob):
        job.add_result("gpt-4", "q1", True, "a", 5.0, 100, ttft=2.0, gen_duration=2.0)
        job.add_result("gpt-4", "q2", False, "b", 3.0, 75, ttft=1.0, gen_duration=1.0)
        # Average of per-response rates: (100/2.0 + 75/1.0) / 2, ignoring took.
        assert job.get_overall_token_rate() == (50.0 + 75.0) / 2

    def test_skips_responses_without_decode_window(self, job: RunJob):
        job.add_result("gpt-4", "q1", True, "a", 5.0, 100, gen_duration=2.0)
        job.add_result("gpt-4", "q2", True, "b", 9.0, 999, gen_duration=0.0)
        assert job.get_overall_token_rate() == 50.0


class TestFailedQuestions:
    def test_label_when_none_failed(self, job: RunJob):
        assert job.failed_label() == "Failed: none"

    def test_mark_and_label(self, job: RunJob):
        job.mark_question_failed(3)
        job.mark_question_failed(1)
        job.mark_question_failed(3)
        assert job.get_failed_questions() == [1, 3]
        assert job.failed_label() == "Failed: 1,3"


class TestReset:
    def test_reset_clears_state(self, job: RunJob):
        job.add_result("gpt-4", "q1", True, "a", 1.0, 50)
        job.mark_question_failed(2)
        job.current_index = 5
        job.reset()
        assert job.current_index == 0
        assert job.results == []
        assert job.failed_questions == []
