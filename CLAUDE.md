# Proje
Domain-specific Bi-Encoder + Cross-Encoder Reranker.
Amazon ESCI (US, İngilizce, e-ticaret). PyTorch + FAISS + FastAPI.
15 günlük solo öğrenme projesi — şu an Gün 4.

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

# Kararlar (verildi, tartışma)
- Shared/siamese encoder (tek self.encoder, iki kez çağrılır)
- Mean pooling (CLS değil)
- L2 normalize + InfoNCE, temperature=0.05
- Her query_id için tek positive → false negative önlemi
- FAISS: IndexFlatIP ile başla

# Sanity check
Eğitimin ilk adımında loss ≈ ln(batch_size) olmalı.
B=16 → 2.77. Çok altındaysa veri sızıntısı, çok üstündeyse kod hatası.

# Takvim
G4: model.py, losses.py, data_prep.py
G5: dataset.py, train.py
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
│   └── train.py
├── scripts/
│   └── data_prep.py
└── data/
```
