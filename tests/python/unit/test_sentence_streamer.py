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

    async def test_does_not_split_common_abbreviation(self):
        streamer = SentenceStreamer(SentenceStreamerConfig(min_dispatch_tokens=2))

        sentences = streamer.push_text("Falei com o dr. Silva ontem. Depois seguimos.")

        assert sentences == ["Falei com o dr. Silva ontem.", "Depois seguimos."]

    async def test_falls_back_to_clause_boundary_for_long_unpunctuated_text(self):
        streamer = SentenceStreamer(
            SentenceStreamerConfig(
                min_dispatch_tokens=4,
                clause_fallback_chars=40,
                max_buffer_chars=80,
                hard_split_min_tokens=10,
            )
        )

        sentences = streamer.push_text(
            "Esta resposta vem longa demais sem ponto final, mas ainda assim precisa sair cedo para a fala continuar fluida"
        )

        assert sentences == ["Esta resposta vem longa demais sem ponto final,"]
        assert streamer.flush() == [
            "mas ainda assim precisa sair cedo para a fala continuar fluida"
        ]
