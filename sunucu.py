from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
from typing import Literal
import psycopg2
import json
import os
import time
import hashlib
from datetime import datetime
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()
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
    "tukenmislik_hissi": "#1c1f24", "ofke_patlamalari": "#2e0f0f", "degersizlik_korkusu": "#101621",
    "anksiyete_krizi": "#191c18", "yalnizlik_ve_izolasyon": "#151515", "onay_ihtiyaci": "#171523",
    "yuzlesme_korkusu": "#221111", "sinir_koyamama": "#1a1e1a", "mukemmeliyetcilik": "#1f2022",
    "zihinsel_asiri_yuklenme": "#161b22", "isyan_ve_haksizlik": "#2c1212", "varolus_sancisi_ve_anlam_arayisi": "#12151e",
    "kurban_psikolojisi": "#1a1614", "kontrol_takintisi": "#17181c", "duygusal_bagimlilik": "#1d151c",
    "olagan_bir_gun": "#0f172a", "varsayilan_karanlik": "#0f172a"
}

KITAPLIK_YOLU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kitaplik.json")
try:
    with open(KITAPLIK_YOLU, "r", encoding="utf-8") as f:
        KITAPLIK = json.load(f)
except FileNotFoundError:
    print("UYARI: kitaplik.json bulunamadı, kitap önerisi verilmeyecek.")
    KITAPLIK = {}

MODEL_ADI = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
YEDEK_MODEL = os.getenv("GEMINI_YEDEK_MODEL")

def gemini_istek_at(icerik_metni, sistem_mesaji, sema, deneme_sayisi=4):
    modeller = [MODEL_ADI] + ([YEDEK_MODEL] if YEDEK_MODEL else [])
    son_hata = None
    for model in modeller:
        for i in range(deneme_sayisi):
            try:
                return client.models.generate_content(
                    model=model,
                    contents=icerik_metni,
                    config=types.GenerateContentConfig(system_instruction=sistem_mesaji, temperature=0.0, top_k=1, response_mime_type="application/json", response_schema=sema)
                )
            except Exception as e:
                son_hata = e
                gecici = "503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e)
                print(f"Gemini hatası ({model}, deneme {i + 1}/{deneme_sayisi}): {e}")
                if gecici and i < deneme_sayisi - 1:
                    time.sleep(2 * (2 ** i))
                    continue
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
    conn.commit()
    conn.close()

init_db()

class GunlukIstegi(BaseModel):
    kullanici_id: int
    metin: str

class KullaniciIstegi(BaseModel):
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
async def kayit_ol(istek: KullaniciIstegi):
    email = istek.email.strip().lower()
    gizli_sifre = sifre_kriptola(istek.sifre)
    
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO kullanicilar (email, sifre) VALUES (%s, %s) RETURNING id", (email, gizli_sifre))
        yeni_id = cursor.fetchone()[0]
        conn.commit()
        conn.close()
        return {"durum": "basarili", "kullanici_id": yeni_id, "email": email}
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
    cursor.execute("SELECT id, email FROM kullanicilar WHERE email = %s AND sifre = %s", (email, gizli_sifre))
    kullanici = cursor.fetchone()
    conn.close()

    if not kullanici:
        raise HTTPException(status_code=400, detail="E-posta adresi veya şifre hatalı!")
        
    return {"durum": "basarili", "kullanici_id": kullanici[0], "email": kullanici[1]}

@app.post("/kaydet")
def kaydet(istek: GunlukIstegi):
    if not istek.metin or len(istek.metin.strip()) < 10:
        raise HTTPException(status_code=400, detail="Metin çok kısa.")

    kullanici_id = istek.kullanici_id
    try:
        bugun = datetime.now().strftime("%Y-%m-%d")
        su_an = datetime.now().strftime("%H:%M:%S")
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO gunluk_metinler (kullanici_id, tarih, saat, metin) VALUES (%s, %s, %s, %s)", (kullanici_id, bugun, su_an, istek.metin))
        
        cursor.execute("SELECT saat, metin FROM gunluk_metinler WHERE tarih = %s AND kullanici_id = %s", (bugun, kullanici_id))
        bugunun_kayitlari = cursor.fetchall()
        conn.commit()
        
        birlestirilmis_metin = "\n\n".join([f"[{k[0]}] {k[1]}" for k in bugunun_kayitlari])
        
        try:
            response = gemini_istek_at(birlestirilmis_metin, SISTEM_MESAJI_5_SUTUN, BesSutunAnaliz)
            analiz = json.loads(response.text)
        except Exception as e:
            print(f"Günlük kaydedildi ama analiz yapılamadı: {e}")
            conn.close()
            return {"durum": "basarili", "analiz": "ertelendi"}
        
        for anahtar, deger in analiz.items():
            if anahtar != "etiket": analiz[anahtar] = robotik_dil_temizle(deger)
        if analiz.get("etiket", "").strip().lower() not in RENK_PALETI:
            analiz["etiket"] = "olagan_bir_gun"

        cursor.execute("DELETE FROM gunluk_icgoruler WHERE tarih = %s AND kullanici_id = %s", (bugun, kullanici_id))
        cursor.execute('''
            INSERT INTO gunluk_icgoruler (
                kullanici_id, tarih, somut_tetikleyici_capasi, vitrin_duygusu_vs_bastirilan_ihtiyac,
                otomatik_savunma_kacis_refleksi, bedensel_ve_enerji_izi, acikta_kalan_yarim_birakilan_dongu, etiket
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            kullanici_id, bugun, analiz.get("somut_tetikleyici_capasi", ""),
            analiz.get("vitrin_duygusu_vs_bastirilan_ihtiyac", ""), analiz.get("otomatik_savunma_kacis_refleksi", ""),
            analiz.get("bedensel_ve_enerji_izi", ""), analiz.get("acikta_kalan_yarim_birakilan_dongu", ""),
            analiz.get("etiket", "")
        ))
        conn.commit()
        conn.close()
        
        # ESKİ HALİ: Ekrana çıktı vermez, sadece başarılı döndürür
        return {"durum": "basarili"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Sunucu meşgul, lütfen tekrar dene.")

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

        yedek_mod = False
        try:
            response = gemini_istek_at(analiz_gecmisi_metni, SISTEM_MESAJI_BUYUK_RESIM, BuyukResimAnaliz)
            buyuk_resim_raporu = json.loads(response.text)
        except Exception as e:
            print(f"Büyük resim analizi yapılamadı, yedek moda geçildi: {e}")
            yedek_mod = True
            buyuk_resim_raporu = {
                "genel_analiz_metni": "Yapay zekâ şu an çok yoğun olduğu için ayrıntılı raporu hazırlayamadım. Son kaydına göre sana bir renk ve kitap seçtim. Birazdan tekrar deneyebilirsin.",
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
