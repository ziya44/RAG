from pathlib import Path

import streamlit as st

from rag_regulations import answer_question_data, ingest

st.set_page_config(page_title="Yönetmelik RAG Asistanı", page_icon="📘", layout="wide")
st.title("📘 Eğitim Yönetmelikleri RAG Asistanı")
st.caption("Sadece yüklediğiniz yönetmelik dosyalarına dayanarak yanıt verir.")

if "docs_path" not in st.session_state:
    st.session_state.docs_path = "data/regulations"
if "index_path" not in st.session_state:
    st.session_state.index_path = "data/index"

with st.sidebar:
    st.header("⚙️ Ayarlar")
    docs_path = st.text_input("Yönetmelik klasörü", st.session_state.docs_path)
    index_path = st.text_input("İndeks klasörü", st.session_state.index_path)
    embedding_model = st.text_input(
        "Embedding modeli",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    llm_model = st.text_input("Yanıt modeli", "Qwen/Qwen2.5-1.5B-Instruct")
    top_k = st.slider("Top-K", min_value=1, max_value=10, value=5)
    min_score = st.slider("Minimum benzerlik skoru", 0.0, 1.0, 0.35, 0.01)

    if st.button("İndeksi Oluştur / Güncelle", type="primary", use_container_width=True):
        try:
            with st.spinner("Dosyalar okunuyor ve indeks oluşturuluyor..."):
                ingest(Path(docs_path), Path(index_path), embedding_model)
            st.success("İndeks başarıyla oluşturuldu.")
        except Exception as exc:
            st.error(f"İndeks oluşturulamadı: {exc}")

st.session_state.docs_path = docs_path
st.session_state.index_path = index_path

st.subheader("❓ Soru Sor")
question = st.text_area(
    "Öğrencinin sorusu",
    placeholder="Örn: Devamsızlık sınırı kaç gündür ve hangi durumda sınıf tekrarı olur?",
    height=120,
)

if st.button("Yanıtla", use_container_width=True):
    if not question.strip():
        st.warning("Lütfen önce bir soru yazın.")
    elif not (Path(index_path) / "regulations.faiss").exists():
        st.warning("Önce soldan 'İndeksi Oluştur / Güncelle' butonuna basın.")
    else:
        try:
            with st.spinner("Kaynaklar aranıyor ve yanıt hazırlanıyor..."):
                result = answer_question_data(
                    index_dir=Path(index_path),
                    question=question,
                    embedding_model=embedding_model,
                    llm_model=llm_model,
                    top_k=top_k,
                    min_score=min_score,
                )

            st.markdown("### Cevap")
            st.write(result["answer"])

            st.markdown("### Kullanılan Kaynaklar")
            if not result["sources"]:
                st.info("Bu soru için eşik üstü kaynak bulunamadı.")
            for i, src in enumerate(result["sources"], start=1):
                with st.expander(
                    f"[{i}] {src['source']} | parça={src['chunk_id']} | skor={src['score']:.3f}"
                ):
                    st.write(src["text"])
        except Exception as exc:
            st.error(f"Yanıt üretilemedi: {exc}")
