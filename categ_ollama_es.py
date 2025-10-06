# -*- coding: utf-8 -*-
"""
categ_ollama_es.py  (v4 - categorías embebidas)
- Clasifica con embeddings de Ollama + refuerzo por palabras clave en español.
- Si Ollama no está disponible: fallback a TF-IDF (scikit-learn) + keywords.
- Si tampoco hay scikit-learn: fallback a solo keywords (sin dependencias).
- Acepta --kw-weight para ajustar el peso de keywords en el score final.
- Las categorías y sus keywords ya NO se leen del Excel: viven en BASE_KW.

Uso:
    python categ_ollama_es.py --in INFILE.xlsx --out OUTFILE.xlsx \
      --sheet-db db --model bge-m3 --otro-th 0.70 --kw-weight 0.30
"""
import argparse, os, re, math, unicodedata, sys
import numpy as np
import pandas as pd

def deaccent(text: str) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = unicodedata.normalize("NFD", text).lower()
    text = "".join([c for c in text if unicodedata.category(c) != "Mn"])
    text = re.sub(r"\s+", " ", text).strip()
    return text

def unique_keep_order(seq):
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

# ---------- CATEGORÍAS Y KEYWORDS EMBEBIDAS ----------
BASE_KW = {
 "Marketing y publicidad": [
   "marketing","publicidad","mercadeo","branding","redes sociales","community manager",
   "campana publicitaria","marketing digital","comunicacion comercial", "Agencia de marketing",
   "relaciones publicas", "ventas", "Agencia creativa con fines comerciales", "Represetante De Ventas",
   "Investigación de mercados", 
 ],
 "Política electoral o partidista": [
   "partido","partido politico","campana","electoral","elecciones","senado","concejo",
   "camara","liberal","conservador","centro democrático","pacto histórico","movimiento politico","campana politica",
    "asamblea departamental", "congreso de la republica",
   "camara de representantes", "senado", "edil", "Ministerio del Interior"
 ],
 "Relaciones internacionales y cooperación internacional": [
   "internacional","relaciones internacionales","cooperacion","ong","oeneg","multilateral","diplomacia",
   "embajada","consulado","acnur", "ACNUR", "onu","oea","usaid","giz","bid","pnud","banco mundial",
   "cooperante","agencia de cooperacion","cooperacion internacional", "ministerio de relaciones exteriores",
   "agencia de las naciones unidas", "organizacion multilateral", "consejo noruego para refugiados", "consejo danes para refugiados",
   "delegacion union europea en colombia", "refugiados", "migraciones", "planificacion de migracion",
   "mision diplomatica", "embajada de noruega", "embajada de suecia", "embajada de dinamarca", "embajada de suiza", "embajada de canada",
   "embajada del reino unido", "embajada de estados unidos", "embajada de francia", "embajada de alemania",
   "embajada de italia", "embajada de espana", "embajada de japon", "embajada de paises bajos", "embajada de belgica", "embajada de australia",
   "embajada de nueva zelanda", "ong", "ayuda humanitaria", "oea", "parlamento andino", " Parlamento de la Comunidad Andina de Naciones", "organizacion de la sociedad civil", "konrad adenauer stiftung",
   "Banco de desarrollo de América Latina", "Naciones Unidas", "Organización de Estados Americanos", "Embajada de Israel", "Embajada de México",
   "Ministerio de Relaciones Exteriores", "Secretario de Relaciones Exteriores", "organismos internacionales", "organismo internacional", "Organismos internacional",
   "Agencia de desarrollo del gobierno frances", "Consulado de Colombia", "Consulado General de Colombia", "embajada de finlandia", "embajada de guatemala", "embajada de honduras", "embajada de nicaragua", "embajada de panama",
   "Embajada de los Estados Unidos", "Embajada Británica en Colombia", "Agencia de la ONU", "Programa Mundial de Alimentos ONU",
   "Cancillería del país", "ProColombia", "Embajada de Colombia en", "Embajada de", "Relaciones Exteriores", "Organización para la Cooperación y Desarrollo Económicos",
   "Agenda America Latina", "Foro Economico Mundial", "Permanent Mission of Colombia"

 ],
 "Consultoría y gestión organizacional": [
   "consultoria","consultor","consultora","gestion organizacional","recursos humanos",
   "rrhh","talento humano","transformacion organizacional"
 ],
 "Ambiente y agricultura": [
   "ambiente","ambiental","medio ambiente","agricola","agro","cambio climatico","tierras",
   "agropecuario","rural","ganaderia","forestal","bosque","biodiversidad","agua","clima",
   "ecologia","pesca","silvicultura","desarrollo rural","agencia nacional de tierras",
   "ministerio de ambiente","ministerio de agricultura","desarrollo agricola","bosques","deforestacion",
   "uso del suelo","fundacion gaia amazonas","fundacion natura","conservacion internacional","wwf",
   "conservacion ambiental", "Corporación Autónoma Regional", "Autoridad Nacional de Licencias Ambientales", 
   "Centro Internacional de Agricultura Tropical", 
 ],
 "Energía y minas": [
   "energia","petroleo","gas","hidrocarburos","mineria","mina","carbon","electricidad","renovable",
   "solar","eolica","hidrica","hidroelectrica","oleoducto","agencia nacional de hidrocarburos",
   "agencia nacional de mineria", "Interconexión Eléctrica", "Ecopetrol", "Empresa petrolera", "Shell"
 ],
 "Software": [
   "software","tecnologia","programador","machine learning","inteligencia artificial","backend","frontend"
 ],
 "Educación y academia": [
   "educacion","docente","profesor","investigacion","universidad","colegio","academia","catedra",
   "estudiante","pedagogia","pedagogico","ministerio de educacion","academia privada",
   "investigador","investigadora", "University", "Gremio universitario", "Colegio privado", "Universidad del Rosario",
   "Universita", "Becas", "Educación Superior", "Institución Educativa", "Institución de Educación Superior", "COLFUTURO",
   "Agencia de educación", " Centro de educación", "Institución de educación privada", "Universidad de carácter privado",
   "Universidad Nacional de Colombia", "Universidad de los Andes", "SENA", "universidad en", "Entidad educativa",
   "Formación técnica y profesional", "Universidad pública", "Institución educativa oficial", "Institución educativa privada",
   "universidad nacional", "Escuela Superior de Administración Pública", "Escuela Tecnológica",
   "Instituto Colombiano de Antropología e Historia"
 ],
 "Turismo y entretenimiento": [
   "turismo","turistico","hotel","hotelero","restaurante","viajes","agencia de viajes","eventos",
   "entretenimiento","ocio","parque","hostal","operador turistico"
 ],
 "Gobierno local, ciudad y territorio": [
   "ciudad","urbano","urbanismo","vivienda","transporte","pot",
   "construccion","inmobiliario","catastro","renovacion urbana","renobo","ordenamiento territorial",
   "captura de valor del suelo","ministerio de vivienda","gestion urbana", "alcaldia", "gobernacion", "gobierno local",
   "entidades territoriales", "asocapitales", "Instituto de Desarrollo Urbano", "desarrollo urbano", "desarrollo territorial",
   "secretaria distrital", "Secretaría Distrital de", "Alcaldia local de", "Alcaldia de", 
   "Empresa de Renovación y Desarrollo Urbano", "Gobernación de", "Instituto Distrital de", "Secretaría de Planeación",
   "Entidad de orden distrital", "Entidad de orden municipal", "Entidad de orden departamental", "Entidad de orden local",
   " Entidad de planeación local", "Secretaría de Planeación", "Alcaldía Municipal de", "Instituto Distrital de la Participación y Acción Comunal",
   "Instituto Distrital", "IDPAC", "políticas públicas", "Entidad administrativa de Medellín",
   "Secretaría De Gobierno", " Entidad del orden distrital", "Superintendencia de Servicios Públicos", "hábitad", "Municipalidad",
   "Habitat y Población"
 ],
 "Gobierno Nacional": [
   "planeacion nacional", "Presidencia de la República", "Departamento Nacional de Planeación", "DNP",
   "Vicepresidencia de la República","unidad nacional de gestion del riesgo", "Rama Ejecutiva del Poder Público",
   "Departamento para la Prosperidad Social", "Vicepresidencia", "Unidad Nacional para la Gestión del Riesgo de Desastres",
   "Unidad Administrativa Especial del orden nacional", "UNGRD", "DPS", " Entidad del orden nacional", "Ministerio de Hacienda",
   "DIAN", "Dirección De Impuestos Y Aduanas Nacionales", "políticas públicas", "Rama Ejecutiva", "Departamento Administrativo Nacional de Estadística"

 ],
 "Cultura y deporte": [
   "cultura","cultural","arte","artes","museo","teatro","musica","deporte","deportiva","entrenador",
   "festival","patrimonio","musica", "intercultural"
 ],
 "Justicia, derecho y órganos de control": [
   "justicia","derecho","legal","juridico","abogado","fiscalia","corte","juzgado","defensoria",
   "procuraduria","notaria","contratacion estatal","litigio","compliance","firma de abogados privada",
   "Bufete de abogados de carácter privado", "entidad judicial", "Auditoría General de la República",
   "Abogado Asociado", "Bufete de abogados", "Consejo de Estado", "Contraloría", "Contraloría General de la República",
    "Corte Constitucional", "Corte Suprema de Justicia", "Defensoría del Pueblo", "Fiscalía General de la Nación",
    "Rama Judicial", "Registraduría Nacional del Estado Civil", "Veeduría Distrital", "Ministerio de Justicia y del Derecho",
    "juridica", 
 ],
 "Medios y comunicación": [
   "medios","comunicacion","periodismo","periodista","radio","television","prensa","editorial","contenido",
   "relaciones publicas","comunicador social","Canal Capital", "Periódico"
 ],
 "Salud y cuidado": [
   "salud","hospital","clinica" ,"farmaceutica","salud publica","atencion primaria en salud",
   "ministerio de salud","pandemia","vacunacion","salud mental","salud sexual y reproductiva","salud comunitaria",
   "cirugias","atencion en salud","prestador de servicios de salud", "tecnología médica", "cuidado infantil", 
   "Primera Infancia", "Instituto colombiano de Bienestar Familiar", "ICBF", " Instituto Nacional de Vigilancia de Medicamentos y Alimentos",
   "Invima" 
 ],
 "Industria y comercio": [
   "gremio","gremial","industria","comercio","camara de comercio","empresarial","pyme",
   "microempresa","microempresas","industrial","manufactura","produccion","fabricante","representante comercial",
   "Sociedad por Acciones Simplificada", "productos alimenticios", "exportaciones", "importaciones", "comercio exterior",
   "Asociación Nacional de Empresarios de Colombia", "Asociación Nacional de Empresarios de Colombia (ANDI)",
   " Empresa de construcción", "Empresa de manufactura", "Gerente", " Empresa de construcción privada",
   "inmobiliario", "Federación de Cámaras de Comercio", "confecamaras", "Supply Chain", "cadena de suministro", "ventas",
   "Empresa multinacional", "Bavaria", "Constructora Bolívar", "British American Tobacco", "Cencosud Colombia", "Empresa privada",  "Empresa de alimentos",
   "Represetante De Ventas", "Empresa de bebidas", "Empresa de servicios", "Inmobiliaria", "Constructora", " Empresa de moda",
   "Fondo Nacional del Ganado", "FEDEGAN", "S.A.", " Corporación privada", "Porcicultura"

 ],
 "Seguridad y fuerzas militares": [
   "policia","fuerza publica","militar","ejercito","armada nacional","fuerza aerea","seguridad",
   "departamento nacional de inteligencia","policia nacional","crimen organizado","inteligencia militar",
   "ministerio de defensa","defensa nacional", "Unidad Nacional de Protección", "Dirección Nacional De Inteligencia",
   "sistema penitenciario", "INPEC", "defensa", "carcelario", "penitenciario"
 ],
 "Paz y reconciliacion": [
   "paz","reconciliacion","conflicto armado","justicia transicional","jep","jurisdiccion especial para la paz",
   "acuerdo final de paz","acuerdo de paz","agencia para la reincorporacion y la normalizacion",
   "comision de la verdad","reparacion","victimas","victima","desminado","reincoporacion",
   "unidad para la atencion y reparacion integral a las victimas", "Agencia Colombiana para la Reintegración",
   "Agencia de Renovación del Territorio", "Agencia Para La Reincorporación Y La Normalización (ARN)", "Posconflicto",
   "Cultura de Paz", "Construcción de Paz", "Alta Consejería Para El Posconflicto", "Centro Nacional de Memoria Histórica",
   "Unidad de Búsqueda de Personas Desaparecidas", "personas dadas por desaparecidas", "desaparecidos", "desaparición forzada",

 ],
 "Transporte y logistica": [
   "transporte","logistica","infraestructura","carretera","transporte publico","transporte de carga","dhl",
   "fedex","aeronautica civil","ministerio de transporte", "Agencia Nacional de Infraestructura", "Aerolinea",
   "Air Europa", "Air France", "Avianca", "Copa Airlines", "Latam", "Wingo", "Supply Chain", "cadeana de suministro",
   "Empresa de mensajería"
 ],
 "Sociedad civil, filantropía y veeduría ciudadana": [
   "sociedad civil","ong", "ayuda humanitaria", "filantropia","responsabilidad social","fundacion corona","bogota como vamos","connect bogota",
   "liderazgo","empoderamiento","fundacion con fines sociales","fundacion grupo social","fundacion karisma","democracia",
   "organizacion de la sociedad civil", "organización sin animo de lucro", "Asociación Cristiana de Jóvenes", " Asociación sin ánimo de lucro",
   "centro de pensamiento", "think tank", "Instituto de Ciencia Política Hernán Echavarría", " Organización religiosa",
   "Conferencia Episcopal de Colombia"
 ],
 "Inversiones, financiero y seguros": [
   "financiero","seguros","banco","bancario","inversion","inversiones","ahorro","credito",
   "riesgo financiero","activos financieros","fondo de pensiones","fondo de cesantias",
   "sociedad administradora de fondos de pensiones", "Compañía de Financiamiento", "Banco AV Villas",
   "Empresa financiera", "Banco de Bogotá", "Banco de Occidente", "Banco Popular", "Banco Caja Social",
   "Investment", "Seguros Bolívar", "entidad financiera", "BBVA", "icetex",
   "Fondo Pasivo Social De Los Ferrocarriles Nacionales De Colombia", "pensiones", "Fondo de Pensiones", "Fondo de Cesantías",
    "Compañía de Seguros", "Entidad Aseguradora", "Entidad de seguros", "colpensiones", "fiduprevisora",
    "Fiduciaria", "FINDETER", "Banco agrario"
  ]
}

def pick_text_cols(df, include=None, exclude=None, exclude_re=None):
    cols = list(df.columns)
    if include:  # whitelist si te interesa
        cols = [c for c in cols if c in include]
    if exclude:
        bad = set(exclude)
        cols = [c for c in cols if c not in bad]
    if exclude_re:
        pat = re.compile(exclude_re, flags=re.I)
        cols = [c for c in cols if not pat.search(c)]
    return cols

def normalize_kw_map(base_kw: dict) -> dict:
    """Devuelve un mapa {Categoria Bonita: [kw_deac, ...]} limpio/normalizado."""
    norm = {}
    for cat, kws in base_kw.items():
        clean_kws = []
        for w in kws:
            w = deaccent(w)
            if w:
                clean_kws.append(w)
        norm[cat] = unique_keep_order(clean_kws)
    return norm

# ---- Backends ----
def embed_ollama(texts, model="bge-m3", url="http://localhost:11434", batch=64, timeout=120):
    import requests
    embs = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        r = requests.post(f"{url}/api/embeddings",
                          json={"model": model, "input": chunk},
                          timeout=timeout)
        if r.status_code == 404:
            raise RuntimeError("OLLAMA_NO_EMBEDDINGS_ENDPOINT")
        r.raise_for_status()
        data = r.json()
        embs.extend([item["embedding"] for item in data["data"]])
    return np.array(embs, dtype="float32")

def cos_sim(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T

def scores_tfidf_keywords(row_texts, categorias, kw_map, otro_th=0.7, kw_weight=0.30):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel
    except Exception:
        return None  # no sklearn
    cat_texts = [deaccent(c) + " " + " ".join(kw_map.get(c, [])) for c in categorias]
    vect = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), norm="l2")
    X = vect.fit_transform(row_texts + cat_texts)
    X_rows = X[:len(row_texts)]
    X_cats = X[len(row_texts):]
    sims = linear_kernel(X_rows, X_cats)
    # plus keywords
    kw_plus = np.zeros_like(sims)
    for j, c in enumerate(categorias):
        kws = kw_map.get(c, [])
        if not kws: continue
        hits = np.array([[1.0 if (w in row_texts[i]) else 0.0 for w in kws] for i in range(len(row_texts))])
        if hits.size:
            raw = hits.sum(axis=1) / max(1.0, math.log(1+len(kws), 2))
            m = raw.max(); kw_plus[:, j] = (raw / m) if m > 0 else 0.0
    w = max(0.0, min(1.0, kw_weight))
    final = (1.0 - w) * sims + w * kw_plus
    # 'Otro'
    idx_otro = categorias.index("Otro")
    no_otro = np.delete(final, idx_otro, axis=1)
    best = no_otro.max(axis=1)
    final[:, idx_otro] = np.maximum(0.0, otro_th - best) / max(1e-9, otro_th)
    return final

def scores_keywords_only(row_texts, categorias, kw_map, otro_th=0.7):
    N, C = len(row_texts), len(categorias)
    S = np.zeros((N, C), dtype="float32")
    for j, c in enumerate(categorias):
        kws = kw_map.get(c, [])
        if not kws: continue
        raw = np.array([sum(1 for w in kws if w and (w in row_texts[i])) for i in range(N)], dtype="float32")
        denom = max(1.0, math.log(1+len(kws), 2))
        raw = raw / denom
        m = raw.max()
        S[:, j] = (raw / m) if m > 0 else 0.0
    idx_otro = categorias.index("Otro")
    no_otro = np.delete(S, idx_otro, axis=1)
    best = no_otro.max(axis=1)
    S[:, idx_otro] = np.maximum(0.0, otro_th - best) / max(1e-9, otro_th)
    return S

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", dest="outfile", required=True)
    ap.add_argument("--sheet-db", default="db")
    # --sheet-cats queda por compatibilidad, pero se ignora:
    ap.add_argument("--sheet-cats", default="(IGNORED)", help="(Ignorado) Las categorías ya viven en el script.")
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--otro-th", type=float, default=0.70)
    ap.add_argument("--kw-weight", type=float, default=0.30, help="Peso (0..1) del plus por keywords sobre la similitud")

    ap.add_argument("--include-cols", nargs="*", default=None)
    ap.add_argument("--exclude-cols", nargs="*", default=None)
    ap.add_argument("--exclude-re", default=None)
    args = ap.parse_args()

    assert os.path.exists(args.infile), f"No existe el archivo: {args.infile}"
    db = pd.read_excel(args.infile, sheet_name=args.sheet_db)

    # categorías desde BASE_KW + 'Otro'
    kw_map = normalize_kw_map(BASE_KW)
    categorias = list(kw_map.keys())
    if "Otro" not in categorias:
        categorias.append("Otro")  # sin keywords, solo umbral
    categorias = unique_keep_order(categorias)
    idx_otro = categorias.index("Otro")

    # textos (todas las columnas)
    # Antes:
# text_cols = db.columns.tolist()

    # Ahora:
    text_cols = pick_text_cols(
        db,
        include=args.include_cols,
        exclude=args.exclude_cols,
        exclude_re=args.exclude_re
    )
    assert len(text_cols) > 0, "No quedaron columnas para categorizar; revisa --include/--exclude."





    row_texts = (
        db.fillna("").astype(str)
          .apply(lambda r: deaccent(" | ".join([f"{c}: {r[c]}" for c in text_cols])), axis=1)
          .tolist()
    )

    # Backend 1: Ollama embeddings
    final = None
    backend = "ollama-embeddings"
    try:
        rows_emb = embed_ollama(row_texts, model=args.model, url=args.ollama_url, batch=args.batch)
        cats_emb = embed_ollama([deaccent(c) for c in categorias], model=args.model, url=args.ollama_url, batch=args.batch)
        sims = cos_sim(rows_emb, cats_emb)
        # plus keywords
        kw_plus = np.zeros_like(sims)
        for j, c in enumerate(categorias):
            kws = kw_map.get(c, [])
            if not kws: continue
            hits = np.array([[1.0 if (w in row_texts[i]) else 0.0 for w in kws] for i in range(len(row_texts))])
            if hits.size:
                raw = hits.sum(axis=1) / max(1.0, math.log(1+len(kws), 2))
                m = raw.max(); kw_plus[:, j] = (raw / m) if m > 0 else 0.0
        w = max(0.0, min(1.0, args.kw_weight))
        final = (1.0 - w) * sims + w * kw_plus
        # 'Otro'
        no_otro = np.delete(final, idx_otro, axis=1)
        best = no_otro.max(axis=1)
        final[:, idx_otro] = np.maximum(0.0, args.otro_th - best) / max(1e-9, args.otro_th)
    except Exception:
        backend = "fallback"

    if final is None:
        # Backend 2: TF-IDF + keywords (si hay scikit-learn)
        final = scores_tfidf_keywords(row_texts, categorias, kw_map, otro_th=args.otro_th, kw_weight=args.kw_weight)
        if final is not None:
            backend = "tfidf+keywords"

    if final is None:
        # Backend 3: Solo keywords
        final = scores_keywords_only(row_texts, categorias, kw_map, otro_th=args.otro_th)
        backend = "keywords-only"

    # Top-3 y export
    scores_df = pd.DataFrame(final, columns=categorias, index=db.index)
    out = db.copy()
    order = np.argsort(-final, axis=1)[:, :3]
    for k in range(order.shape[1]):
        out[f"Top{k+1}_Categoria"] = [categorias[j] for j in order[:,k]]
        out[f"Top{k+1}_Score"] = [float(final[i, order[i,k]]) for i in range(final.shape[0])]

    with pd.ExcelWriter(args.outfile, engine="openpyxl") as w:
        out.to_excel(w, index=False, sheet_name="top3")
        scores_df.to_excel(w, index=False, sheet_name="scores")
        pd.DataFrame([{
            "filas_procesadas": int(len(db)),
            "categorias_detectadas": ", ".join(categorias),
            "backend_usado": backend,
            "modelo_embeddings": args.model if backend=="ollama-embeddings" else "",
            "umbral_otro": float(args.otro_th),
            "peso_keywords": float(args.kw_weight),
            "nota": "Categorías embebidas en el script; --sheet-cats ignorado."
        }]).to_excel(w, index=False, sheet_name="reporte")

    print("Listo:", args.outfile, "| backend:", backend)

if __name__ == "__main__":
    main()
