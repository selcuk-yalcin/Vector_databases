# 🔒 Restart Koruma Rehberi

## 🎯 Sorun

Her notebook restart'ında:
- ❌ Pinecone index'leri silinip yeniden oluşturuluyor
- ❌ Tüm dökümanlar tekrar embedding'leniyor (2-5 dakika)
- ❌ OpenAI API kredisi gereksiz yere kullanılıyor
- ❌ Zaman kaybı

## ✅ Çözüm

Tüm "silme" ve "yükleme" kodları hashtag (#) ile yorum satırına alındı.

---

## 📋 Korunan Hücreler

### 1️⃣ **Hücre 3: Pinecone Index Oluşturma**
```python
# ⚠️ RESTART KORUMASI: Aşağıdaki kodlar hashtag'li
# Index zaten varsa ÇALIŞTIRMAYIN

# if INDEX_NAME in pc.list_indexes().names():
#     pc.delete_index(INDEX_NAME)

# pc.create_index(...)
```

**Ne Yapar:**
- ✅ Mevcut index'e bağlanır (silmez)
- ⚡ Anında çalışır

---

### 2️⃣ **Hücre 6: Döküman Yükleme**
```python
# ⚠️ RESTART KORUMASI: Döküman yükleme hashtag'li

# index = VectorStoreIndex.from_documents(...)  # Yeniden yükleme YAPILMAZ

# Alternatif: Mevcut index'e bağlan
index = VectorStoreIndex.from_vector_store(vector_store)
```

**Ne Yapar:**
- ✅ Mevcut dökümanları kullanır
- ⚡ Yeniden yükleme yapmaz

---

### 3️⃣ **Hücre 11: Hybrid Index Oluşturma**
```python
# ⚠️ RESTART KORUMASI: Hybrid index hashtag'li

# if HYBRID_INDEX_NAME in pc.list_indexes().names():
#     pc.delete_index(HYBRID_INDEX_NAME)

# pc.create_index(...)
```

**Ne Yapar:**
- ✅ Mevcut hybrid index'e bağlanır
- ⚡ Yeniden oluşturmaz

---

### 4️⃣ **Hücre 13: Hybrid Vektör Yükleme**
```python
# ⚠️ RESTART KORUMASI: En uzun işlem hashtag'li (2-5 dakika)

# for i, doc in enumerate(tqdm(documents)):
#     dense_vector = embed_model.get_text_embedding(doc.text)
#     sparse_vector = bm25_encoder.encode_documents([doc.text])[0]
#     hybrid_index.upsert(...)

# Alternatif: Mevcut index'e bağlan
hybrid_index = pc.Index(HYBRID_INDEX_NAME)
```

**Ne Yapar:**
- ✅ Mevcut hybrid vektörleri kullanır
- ⚡ 2-5 dakikalık işlemi atlar
- 💰 OpenAI API kredisi harcamaz

---

## 🔓 İLK KURULUM (Hashtag'leri Kaldırın)

Aşağıdaki durumlarda hashtag'leri **KALDIRSMANIZ** gerekir:

### 1. İlk Defa Kurulum
```python
# Tüm hashtag'leri kaldır ve sırayla çalıştır:
# - Hücre 3: Index oluştur
# - Hücre 6: Dökümanları yükle
# - Hücre 11: Hybrid index oluştur
# - Hücre 13: Hybrid vektörleri yükle
```

### 2. Yeni Döküman Ekleme
```python
# Sadece şunları çalıştır:
# - Hücre 6: Dökümanları yeniden yükle
# - Hücre 13: Hybrid vektörleri yeniden yükle
```

### 3. Tamamen Sıfırdan Başlama
```python
# Tüm hashtag'leri kaldır ve sıfırdan başla
```

---

## ⚡ RESTART SONRASI (Hashtag'ler Olduğu Gibi)

Her restart'ta çalıştırmanız gerekenler:

```python
# 1. Hücre 1-2: Import'lar ve API key'ler ✅
# 2. Hücre 3: Mevcut index'e bağlan (hashtag'li) ✅
# 3. Hücre 4: Dökümanları parse et ✅
# 4. Hücre 5: Embedding model ayarla ✅
# 5. Hücre 6: Mevcut index'e bağlan (hashtag'li) ✅
# 6. Hücre 7-9: LLM ve prompt ayarları ✅
# 7. Hücre 10-12: BM25 encoder eğitimi ✅
# 8. Hücre 11: Mevcut hybrid index'e bağlan (hashtag'li) ✅
# 9. Hücre 13: Mevcut hybrid vektörlere bağlan (hashtag'li) ✅
# 10. Hücre 14-18: Test soruları ✅
```

**Süre:** ~30 saniye (2-5 dakika yerine!)

---

## 🎯 Özet

| Durum | Hashtag'leri Kaldır? | Süre |
|-------|----------------------|------|
| **İlk Kurulum** | ✅ EVET | 2-5 dakika |
| **Her Restart** | ❌ HAYIR | 30 saniye |
| **Yeni Döküman** | ✅ EVET (sadece upload hücreleri) | 1-2 dakika |
| **Sıfırdan Başla** | ✅ EVET (tümü) | 2-5 dakika |

---

## 💡 Faydalar

### Öncesi (Hashtag'siz)
- ❌ Her restart: 2-5 dakika
- ❌ Gereksiz API kullanımı
- ❌ Gereksiz index silme/oluşturma

### Sonrası (Hashtag'li)
- ✅ Her restart: 30 saniye
- ✅ Sıfır gereksiz API kullanımı
- ✅ Mevcut index'leri kullan
- ✅ Hızlı geliştirme

---

## ⚠️ Önemli Notlar

1. **BM25 Encoder:** Her restart'ta yeniden eğitilmesi gerekir (hızlı, ~5 saniye)
2. **Döküman Parse:** Her restart'ta yeniden parse edilmesi gerekir (hızlı, ~2 saniye)
3. **Index Connection:** Hashtag'li kod otomatik olarak mevcut index'e bağlanır
4. **API Key'ler:** Her restart'ta `.env` dosyasından yüklenir

---

## 🔧 Manuel Kontrol

Pinecone dashboard'undan index'lerin var olduğunu kontrol edin:
```
https://app.pinecone.io/

Index'ler:
- isg-rag-openai-3072 (cosine) ✅
- isg-hybrid-openai-3072 (dotproduct) ✅
```

---

**Güncelleme Tarihi:** 17 Ocak 2026  
**Durum:** ✅ Restart korumalı sistem aktif
