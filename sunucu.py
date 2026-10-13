from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
from typing import Literal
import psycopg2
import json
import os
import time
import hashlib
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

# Render sunucuları UTC çalışır; günün Türkiye saatine (UTC+3) göre bitmesi için
TR_SAAT = timezone(timedelta(hours=3))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI()

SISTEM_MESAJI_5_SUTUN = """Sen Mentalist'sin. Kullanıcı sana gün içinde parça parça yazdığı TÜM günlük metinlerinin birleştirilmiş halini verecek.
Görev: Bu toplam bağlamı inceleyip, kullanıcının BUGÜNKÜ zihinsel haritasını tek bir 5 sütunda özetlemek.
ÇOK ÖNEMLİ KURAL: Her sütunu MAKSİMUM 1 VEYA 2 CÜMLE ile, nokta atışı ve çok kısa yaz.
ASLA 3. şahıs veya robotik diller kullanma. 'Sen' dili kullan."""

SISTEM_MESAJI_BUYUK_RESIM = """Sen Mentalist'sin. Sana kullanıcının geçmiş günlere ait '5 Sütunluk' bilinçaltı verileri verilecek.
Görev: Bu verileri birleştirerek olay, bağlam, vitrin duygu ve bastırılan asıl duygu arasındaki ilişkiyi deşifre eden somut, şefkatli ama yüzleştirici bir 'Zihinsel Örüntü Raporu' yazmak.

ÇOK ÖNEMLİ KURAL: 'baskin_etiket' seçimi yaparken, kullanıcının EN SON (en güncel) tarihte yaşadığı duygu durumuna daha büyük bir ağırlık ver. Eski konuları kök sebep olarak diretme.
Lafı uzatma, doğrudan analizini sun. ASLA 3. şahıs kullanma."""

EtiketTipi = Literal[
    "tukenmislik_hissi", "ofke_patlamalari", "degersizlik_korkusu", "anksiyete_krizi",
    "yalnizlik_ve_izolasyon", "onay_ihtiyaci", "yuzlesme_korkusu", "sinir_koyamama",
    "mukemmeliyetcilik", "zihinsel_asiri_yuklenme", "isyan_ve_haksizlik",
    "varolus_sancisi_ve_anlam_arayisi", "kurban_psikolojisi", "kontrol_takintisi",
    "duygusal_bagimlilik", "olagan_bir_gun"
]

class BesSutunAnaliz(BaseModel):
    somut_tetikleyici_capasi: str
    vitrin_duygusu_vs_bastirilan_ihtiyac: str
    otomatik_savunma_kacis_refleksi: str
    bedensel_ve_enerji_izi: str
    acikta_kalan_yarim_birakilan_dongu: str
    etiket: EtiketTipi

class BuyukResimAnaliz(BaseModel):
    genel_analiz_metni: str
    baskin_etiket: EtiketTipi

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

RENK_PALETI = {
    "tukenmislik_hissi": "#D6DCE4",
    "ofke_patlamalari": "#F5C8C4",
    "degersizlik_korkusu": "#CDE4F0",
    "anksiyete_krizi": "#E4D9F0",
    "yalnizlik_ve_izolasyon": "#D2E8E4",
    "onay_ihtiyaci": "#F6E0C6",
    "yuzlesme_korkusu": "#EBD3DC",
    "sinir_koyamama": "#DCEBD0",
    "mukemmeliyetcilik": "#D6DDF0",
    "zihinsel_asiri_yuklenme": "#F2E8C2",
    "isyan_ve_haksizlik": "#F4D2B8",
    "varolus_sancisi_ve_anlam_arayisi": "#D8D3EA",
    "kurban_psikolojisi": "#E3D5CC",
    "kontrol_takintisi": "#E0DDD3",
    "duygusal_bagimlilik": "#F0D0E4",
    "olagan_bir_gun": "#E4F0E0",
    "varsayilan_karanlik": "#E4F0E0"
}

KITAPLIK_YOLU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kitaplik.json")
try:
    with open(KITAPLIK_YOLU, "r", encoding="utf-8") as f:
        KITAPLIK = json.load(f)
except FileNotFoundError:
    print("UYARI: kitaplik.json bulunamadı, kitap önerisi verilmeyecek.")
    KITAPLIK = {}

MODEL_ADI = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# Virgülle ayrılmış yedek model listesi. Örn: GEMINI_YEDEK_MODELLER=model-a,model-b
YEDEK_MODELLER = [m.strip() for m in (os.getenv("GEMINI_YEDEK_MODELLER") or os.getenv("GEMINI_YEDEK_MODEL") or "").split(",") if m.strip()]
# 1 ise, listede olmayan ama hesabında kullanılabilen "flash" modelleri de son çare olarak denenir
OTOMATIK_YEDEK = os.getenv("GEMINI_OTOMATIK_YEDEK", "1") == "1"
MAKS_MODEL_SAYISI = 5
# Google'ın model listesinden (ai.google.dev) alınan hazır yedekler. Kodda hazır olduğu için ayrıca bir şey yazman gerekmez.
VARSAYILAN_YEDEKLER = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite"]

_bloklu_modeller = {}  # model -> tekrar denenebileceği zaman (epoch saniye)
_kesfedilen = {"zaman": 0, "liste": []}

def _model_blokla(model, saniye):
    _bloklu_modeller[model] = time.time() + saniye

def _model_bloklu_mu(model):
    return _bloklu_modeller.get(model, 0) > time.time()

def kesfedilen_modeller():
    """Hesapta kullanılabilen metin modellerini (flash ailesi) bulur, 6 saat önbellekte tutar."""
    if time.time() - _kesfedilen["zaman"] < 6 * 3600:
        return _kesfedilen["liste"]
    _kesfedilen["zaman"] = time.time()
    try:
        adaylar = []
        yasakli = ("image", "tts", "audio", "live", "embedding", "robotics", "computer", "native", "exp", "vision")
        for m in client.models.list():
            ad = m.name.replace("models/", "")
            eylemler = getattr(m, "supported_actions", None) or []
            if eylemler and "generateContent" not in eylemler:
                continue
            kucuk = ad.lower()
            if "flash" not in kucuk or any(k in kucuk for k in yasakli):
                continue
            adaylar.append(ad)
        adaylar.sort(reverse=True)                      # yeni sürümler önde
        adaylar.sort(key=lambda a: "lite" in a.lower()) # lite modeller en sonda
        _kesfedilen["liste"] = adaylar
    except Exception as e:
        print(f"Model listesi alınamadı: {e}")
    return _kesfedilen["liste"]

def model_zinciri():
    zincir = []
    for m in [MODEL_ADI] + YEDEK_MODELLER + VARSAYILAN_YEDEKLER + (kesfedilen_modeller() if OTOMATIK_YEDEK else []):
        if m not in zincir:
            zincir.append(m)
    return zincir

def gemini_istek_at(icerik_metni, sistem_mesaji, sema, deneme_sayisi=3):
    tum_zincir = model_zinciri()
    zincir = [m for m in tum_zincir if not _model_bloklu_mu(m)][:MAKS_MODEL_SAYISI]
    if not zincir:
        raise RuntimeError("429 RESOURCE_EXHAUSTED: Tüm modellerin kotası dolu ya da modellere ulaşılamıyor.")

    son_hata = None
    for model in zincir:
        deneme = deneme_sayisi if model == MODEL_ADI else 2
        for i in range(deneme):
            try:
                yanit = client.models.generate_content(
                    model=model,
                    contents=icerik_metni,
                    config=types.GenerateContentConfig(system_instruction=sistem_mesaji, temperature=0.0, top_k=1, response_mime_type="application/json", response_schema=sema)
                )
                if model != MODEL_ADI:
                    print(f"Yedek model kullanıldı: {model}")
                return yanit
            except Exception as e:
                son_hata = e
                hata = str(e)
                print(f"Gemini hatası ({model}, deneme {i + 1}/{deneme}): {hata[:300]}")

                gunluk_kota = "429" in hata and "PerDay" in hata
                gecici = ("503" in hata or "UNAVAILABLE" in hata or "429" in hata) and not gunluk_kota

                if gunluk_kota:
                    _model_blokla(model, 30 * 60)       # kota dolu: 30 dk atla
                    break
                if "404" in hata or "NOT_FOUND" in hata:
                    _model_blokla(model, 24 * 3600)     # model yok: sunucu yeniden başlayana dek atla
                    break
                if "400" in hata or "INVALID_ARGUMENT" in hata:
                    _model_blokla(model, 10 * 60)       # bu model isteği/şemayı desteklemiyor olabilir
                    break
                if gecici and i < deneme - 1:
                    time.sleep(2 * (2 ** i))
                    continue
                if gecici:
                    _model_blokla(model, 30)            # yoğunluk: kısa süre atla
                break
    raise son_hata

def get_db():
    return psycopg2.connect(DATABASE_URL)

def sifre_kriptola(sifre: str):
    return hashlib.sha256(sifre.encode()).hexdigest()

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS kullanicilar (
                      id SERIAL PRIMARY KEY, 
                      email TEXT UNIQUE, 
                      sifre TEXT)''')
    cursor.execute("ALTER TABLE kullanicilar ADD COLUMN IF NOT EXISTS ad TEXT")
    cursor.execute("ALTER TABLE kullanicilar ADD COLUMN IF NOT EXISTS soyad TEXT")

    cursor.execute('''CREATE TABLE IF NOT EXISTS gunluk_metinler (
                      id SERIAL PRIMARY KEY, 
                      kullanici_id INTEGER, 
                      tarih TEXT, 
                      saat TEXT, 
                      metin TEXT)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS gunluk_icgoruler (
                      kullanici_id INTEGER, 
                      tarih TEXT, 
                      somut_tetikleyici_capasi TEXT, 
                      vitrin_duygusu_vs_bastirilan_ihtiyac TEXT, 
                      otomatik_savunma_kacis_refleksi TEXT, 
                      bedensel_ve_enerji_izi TEXT, 
                      acikta_kalan_yarim_birakilan_dongu TEXT, 
                      etiket TEXT,
                      PRIMARY KEY (kullanici_id, tarih))''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS genel_analiz_onbellek (
                      kullanici_id INTEGER PRIMARY KEY,
                      imza TEXT,
                      analiz TEXT,
                      etiket TEXT)''')
    conn.commit()
    conn.close()

init_db()

class GunlukIstegi(BaseModel):
    kullanici_id: int
    metin: str

class KullaniciIstegi(BaseModel):
    email: EmailStr
    sifre: str

class KayitIstegi(BaseModel):
    ad: str
    soyad: str
    email: EmailStr
    sifre: str

def robotik_dil_temizle(metin: str) -> str:
    if not isinstance(metin, str): return metin
    return metin.replace("Karakterin", "Senin").replace("Karakter", "Sen").replace("Kullanıcının", "Senin").replace("Kullanıcı", "Sen").replace("Yazar", "Sen")

@app.get("/")
def ana_sayfa():
    return FileResponse("index.html") if os.path.exists("index.html") else {"mesaj": "Bulunamadı."}

@app.get("/patrick_jane.jpg")
def arka_plan_resmi():
    return FileResponse("patrick_jane.jpg") if os.path.exists("patrick_jane.jpg") else {"mesaj": "Resim bulunamadı."}

@app.post("/kayit_ol")
async def kayit_ol(istek: KayitIstegi):
    ad = istek.ad.strip()[:60]
    soyad = istek.soyad.strip()[:60]
    email = istek.email.strip().lower()

    if not ad or not soyad:
        raise HTTPException(status_code=400, detail="Ad ve soyad boş bırakılamaz.")
    if len(istek.sifre) < 6:
        raise HTTPException(status_code=400, detail="Şifre en az 6 karakter olmalı.")

    gizli_sifre = sifre_kriptola(istek.sifre)

    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO kullanicilar (email, sifre, ad, soyad) VALUES (%s, %s, %s, %s) RETURNING id", (email, gizli_sifre, ad, soyad))
        yeni_id = cursor.fetchone()[0]
        conn.commit()
        conn.close()
        return {"durum": "basarili", "kullanici_id": yeni_id, "email": email, "ad": ad}
    except psycopg2.IntegrityError:
        conn.rollback()
        conn.close()
        raise HTTPException(status_code=400, detail="Bu e-posta adresi sistemde zaten kayıtlı!")

@app.post("/giris_yap")
async def giris_yap(istek: KullaniciIstegi):
    email = istek.email.strip().lower()
    gizli_sifre = sifre_kriptola(istek.sifre)

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, ad FROM kullanicilar WHERE email = %s AND sifre = %s", (email, gizli_sifre))
    kullanici = cursor.fetchone()
    conn.close()

    if not kullanici:
        raise HTTPException(status_code=400, detail="E-posta adresi veya şifre hatalı!")

    return {"durum": "basarili", "kullanici_id": kullanici[0], "email": kullanici[1], "ad": kullanici[2] or ""}

@app.post("/kaydet")
def kaydet(istek: GunlukIstegi, arka_plan: BackgroundTasks):
    if not istek.metin or len(istek.metin.strip()) < 10:
        raise HTTPException(status_code=400, detail="Metin çok kısa.")

    kullanici_id = istek.kullanici_id
    try:
        bugun = datetime.now(TR_SAAT).strftime("%Y-%m-%d")
        su_an = datetime.now(TR_SAAT).strftime("%H:%M:%S")

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO gunluk_metinler (kullanici_id, tarih, saat, metin) VALUES (%s, %s, %s, %s)", (kullanici_id, bugun, su_an, istek.metin))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Kayıt hatası: {e}")
        raise HTTPException(status_code=500, detail="Sunucu meşgul, lütfen tekrar dene.")

    # Cevap hemen döner; Gemini analizi kullanıcıyı bekletmeden arka planda yapılır
    arka_plan.add_task(gunluk_icgorusu_uret, kullanici_id, bugun)
    return {"durum": "basarili"}

@app.get("/genel_analiz/{kullanici_id}")
def genel_analiz(kullanici_id: int):
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT tarih, somut_tetikleyici_capasi, vitrin_duygusu_vs_bastirilan_ihtiyac, otomatik_savunma_kacis_refleksi, etiket FROM gunluk_icgoruler WHERE kullanici_id = %s ORDER BY tarih ASC", (kullanici_id,))
        gecmis_veriler = cursor.fetchall()
        conn.close()

        if not gecmis_veriler:
            return {"analiz": "Henüz sana ait yeterli veri yok.", "renk": RENK_PALETI["varsayilan_karanlik"], "kitap": None}

        analiz_gecmisi_metni = "Kullanıcının Zihinsel Örüntüleri:\n"
        for v in gecmis_veriler:
            analiz_gecmisi_metni += f"Tarih: {v[0]} | Tetikleyici: {v[1]} | Duygu Çatışması: {v[2]} | Savunma: {v[3]} | Etiket: {v[4]}\n"

        # Veriler değişmediyse Gemini'yi tekrar çağırma: kayıtlı raporu göster (kota ve süre tasarrufu)
        imza = hashlib.sha256(analiz_gecmisi_metni.encode()).hexdigest()
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT imza, analiz, etiket FROM genel_analiz_onbellek WHERE kullanici_id = %s", (kullanici_id,))
        onbellek = cursor.fetchone()
        conn.close()

        if onbellek and onbellek[0] == imza:
            etiket = onbellek[2] if onbellek[2] in RENK_PALETI else "olagan_bir_gun"
            return {"analiz": onbellek[1], "renk": RENK_PALETI[etiket], "kitap": KITAPLIK.get(etiket), "yedek": False}

        yedek_mod = False
        try:
            response = gemini_istek_at(analiz_gecmisi_metni, SISTEM_MESAJI_BUYUK_RESIM, BuyukResimAnaliz)
            buyuk_resim_raporu = json.loads(response.text)

            temiz_etiket = buyuk_resim_raporu.get("baskin_etiket", "").strip().lower()
            if temiz_etiket not in RENK_PALETI: temiz_etiket = "olagan_bir_gun"
            temiz_metin = robotik_dil_temizle(buyuk_resim_raporu.get("genel_analiz_metni", ""))

            conn = get_db()
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO genel_analiz_onbellek (kullanici_id, imza, analiz, etiket)
                              VALUES (%s, %s, %s, %s)
                              ON CONFLICT (kullanici_id) DO UPDATE
                              SET imza = EXCLUDED.imza, analiz = EXCLUDED.analiz, etiket = EXCLUDED.etiket''',
                           (kullanici_id, imza, temiz_metin, temiz_etiket))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Büyük resim analizi yapılamadı, yedek moda geçildi: {e}")
            yedek_mod = True
            kota_doldu = "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)

            if onbellek:
                # Eski ama gerçek bir rapor varsa onu göster
                buyuk_resim_raporu = {
                    "genel_analiz_metni": onbellek[1] + "\n\n(Not: Yapay zekâ şu an yeni rapor üretemediği için en son hazırlanan rapor gösteriliyor.)",
                    "baskin_etiket": onbellek[2] or "olagan_bir_gun",
                }
            else:
                neden = ("Yapay zekâ servisinin günlük kullanım sınırı doldu" if kota_doldu else "Yapay zekâ şu an çok yoğun")
                buyuk_resim_raporu = {
                    "genel_analiz_metni": neden + ", bu yüzden ayrıntılı raporu hazırlayamadım. Son kaydına göre sana bir renk ve kitap seçtim. Daha sonra tekrar deneyebilirsin.",
                    "baskin_etiket": gecmis_veriler[-1][4] or "olagan_bir_gun",
                }
        baskin_etiket = buyuk_resim_raporu.get("baskin_etiket", "").strip().lower()
        if baskin_etiket not in RENK_PALETI: baskin_etiket = "olagan_bir_gun"

        analiz_metni = robotik_dil_temizle(buyuk_resim_raporu.get("genel_analiz_metni", ""))
        arka_plan_rengi = RENK_PALETI.get(baskin_etiket, RENK_PALETI["varsayilan_karanlik"])
        
        kitap_onerisi = KITAPLIK.get(baskin_etiket)

        return {"analiz": analiz_metni, "renk": arka_plan_rengi, "kitap": kitap_onerisi, "yedek": yedek_mod}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Sunucu meşgul, lütfen tekrar dene.")
