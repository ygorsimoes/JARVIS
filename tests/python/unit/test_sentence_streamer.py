import pytest

from jarvis.core.sentence_streamer import SentenceStreamer, SentenceStreamerConfig


@pytest.mark.asyncio
class TestSentenceStreamer:
    async def test_hard_boundary_dispatches_sentence(self):
        streamer = SentenceStreamer(SentenceStreamerConfig(min_dispatch_tokens=3))

        async def token_stream():
            for chunk in ["Isto ", "e ", "um teste."]:
                yield chunk

        sentences = await streamer.collect(token_stream())
        assert sentences == ["Isto e um teste."]

    async def test_soft_boundary_requires_length(self):
        streamer = SentenceStreamer(
            SentenceStreamerConfig(min_dispatch_tokens=2, min_soft_boundary_chars=20)
        )

        short = streamer.push_text("Resumo curto:")
        long = streamer.push_text(" detalhes suficientes para fechar:")

        assert short == []
        assert long == ["Resumo curto: detalhes suficientes para fechar:"]

    async def test_flush_emits_remainder(self):
        streamer = SentenceStreamer(SentenceStreamerConfig(min_dispatch_tokens=8))
        streamer.push_text("Sem pontuacao final")
        assert streamer.flush() == ["Sem pontuacao final"]

    async def test_multiple_sentences_are_split_incrementally(self):
        streamer = SentenceStreamer(SentenceStreamerConfig(min_dispatch_tokens=2))
        sentences = streamer.push_text(
            "Primeira frase completa. Segunda frase completa."
        )
        assert sentences == ["Primeira frase completa.", "Segunda frase completa."]
