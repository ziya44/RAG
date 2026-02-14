# Eğitim Yönetmelikleri için RAG (Retrieval-Augmented Generation)

Bu depo, eğitim-öğretim yönetmeliklerinden beslenen ve yalnızca bu kaynaklara dayanarak cevap veren bir RAG örneği içerir.

## Hedef
- Sadece verilen yönetmelikleri bilgi kaynağı olarak kullanmak.
- Kaynak dışı sorularda "yönetmeliklerde bulunamadı" diyebilmek.
- Her cevapta ilgili metin parçalarını alıntılamak.

## Hızlı başlangıç

### 1) Kurulum
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Veri klasörünü hazırlama
Yönetmelik dosyalarını `data/regulations` klasörüne koyun.
Desteklenen formatlar:
- `.txt`
- `.md`
- `.pdf`

### 3) Vektör indeksini oluşturma
```bash
python rag_regulations.py ingest --source-dir data/regulations --index-dir data/index
```

### 4) Soru sorma
```bash
python rag_regulations.py ask --index-dir data/index --question "Devamsızlık sınırı nedir?"
```

## Neden bu yaklaşım güvenli?
- Cevap üretiminden önce sadece yönetmeliklerden ilgili pasajlar çekilir.
- Benzerlik skoru eşiği altında sonuç varsa model cevap vermez.
- Prompt, modeli "yalnızca bağlama dayan" kuralına zorlar.
- Cevap sonuna kullanılan pasajların dosya adı ve parça numarası eklenir.

## Önerilen üretim iyileştirmeleri
- OCR gereken taranmış PDF’ler için OCR hattı ekleyin.
- Metadatalara tarih/sürüm bilgisi ekleyin.
- Eski yönetmelikleri arşivleyip etkin sürüm filtresi uygulayın.
- API + kimlik doğrulama ile öğrenci kullanımına açın.

## Uyarı
Bu örnek bir başlangıçtır. Kurumsal kullanımda hukuk/ölçme-değerlendirme ekipleriyle doğrulama süreci kurulmalıdır.
