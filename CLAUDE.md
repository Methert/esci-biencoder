# Proje
Domain-specific Bi-Encoder + Cross-Encoder Reranker.
Amazon ESCI (US, İngilizce, e-ticaret). PyTorch + FAISS + FastAPI.
15 günlük solo öğrenme projesi — şu an Gün 6 (G4-5 tamam).

# Çalışma şekli — ÖNEMLİ
Bu bir öğrenme projesi. Kullanıcı 4. sınıf Bilgisayar Mühendisliği
öğrencisi; matematik ve klasik ML güçlü, PyTorch deneyimi yeni.

- Hazır kod bloğu yığma. Önce mekanizmayı sor/açıkla, sonra kullanıcıdan
  ilgili fonksiyonu yazmasını iste, sonra review et.
- Bir seferde bir modül. Tüm dosyaları birden üretme.
- Token verimli ol, gereksiz açıklama yapma.
- Kullanıcı yazdıkça refactor öner, ama önce kendi denemesini iste.
- Faz geçişlerinde 5-6 maddelik PROJECT_STATE özeti çıkar.

# Donanım
RTX 3060 Laptop, 6.4 GB VRAM.
Başlangıç: bge-small-en-v1.5, batch=16, max_seq_len=128, AMP açık.
Ölçülen (B=16): peak VRAM 0.89 GB, ~9.4 it/s → 1 epoch (~3k adım) ≈ 5 dk.

# Ortam
- Python: `C:\Users\Excalibur\anaconda3\envs\ML\python.exe` (torch+transformers sadece bu env'de)
- Komutlar repo kökünden: `python scripts\data_prep.py`, `python src\train.py [--max_steps 20]`
- ESCI (`tasksource/esci`) ve model ağırlıkları HF cache'te → `HF_HUB_OFFLINE=1` ile çalışır
- ESCI değerleri: `product_locale` ∈ {us, es, jp}; `esci_label` ∈ {Exact, Substitute, Complement, Irrelevant} (tek harf DEĞİL)

# Kararlar (verildi, tartışma)
- Shared/siamese encoder (tek self.encoder, iki kez çağrılır)
- Mean pooling (CLS değil)
- L2 normalize + InfoNCE, temperature=0.05
- Her query_id için tek positive → false negative önlemi
- query ve product_title metinleri dedup (aynı metin batch'te false negative olur)
- Train/val: ESCI train split'ten 50k çift, %5 val (query bazında ayrık). ESCI test split → G7 eval
- FAISS: IndexFlatIP ile başla

# Açık kararlar (G6 öncesi)
- Batch boyutu: VRAM'in %14'ü kullanılıyor; 64/128 daha çok in-batch negatif demek (lr de gözden geçirilmeli)

# Sanity check
Rastgele init modelde ilk adım loss ≈ ln(batch_size) (B=16 → 2.77). Bu, forward/loss kodunu doğrular
(rastgele BERT ile ölçüldü: 2.79).
Pretrained bge'de ilk adım loss çok daha düşük olur (ölçülen 0.08) — bu sızıntı DEĞİL, model zaten retrieval
için eğitilmiş. Bu modelde kontrol edilecekler:
- Zero-shot val baseline kaydedilir; fine-tune sonrası val bunu geçmeli
- Train/val query kesişimi boş olmalı (sızıntı kontrolü)
- In-batch acc@1 B=16'da tavana yakın (0.957) → asıl ölçüt G7'deki tam korpus Recall/nDCG

# Baseline (B=16, in-batch, val=2.5k)
- Zero-shot: val_loss 0.258, acc@1 0.957
- 20 adım smoke run sonrası: val_loss 0.206, acc@1 0.959

# Takvim
G4: model.py, losses.py, data_prep.py ✓
G5: dataset.py, train.py ✓ (20 adım smoke run geçti)
G6: ilk eğitim koşusu (hard deadline)
G7: eval (Recall@100, MRR, nDCG@10)
G8-9: FAISS | G10-11: cross-encoder | G12-13: FastAPI | G14-15: tampon

# Repo iskeleti
```
esci-biencoder/
├── CLAUDE.md
├── requirements.txt
├── src/
│   ├── model.py
│   ├── losses.py
│   ├── dataset.py
│   ├── train.py
│   └── test/smoke_test.py
├── scripts/
│   └── data_prep.py
├── data/          (gitignore: train/val.parquet)
└── checkpoints/   (gitignore: best.pt)
```
