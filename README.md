# IA para Inclusión · Reconocimiento de señas LSC

Aplicación de escritorio para **grabar, entrenar y reconocer en tiempo real gestos de
la Lengua de Señas Colombiana (LSC)** usando la cámara web. Está pensada para
actividades educativas: cada grupo o persona graba sus propios ejemplos de tres
gestos, entrena un modelo pequeño en su computador y ve cómo la IA los reconoce.

No se guarda video ni imágenes. La aplicación solo almacena **puntos geométricos
de la mano**, cifrados con una contraseña que elige cada grupo.

---

## Contenido

1. [Requisitos](#requisitos)
2. [Instalación](#instalación)
3. [Uso](#uso)
4. [Privacidad y seguridad](#privacidad-y-seguridad)
5. [Cómo funciona](#cómo-funciona)
6. [Estructura del proyecto](#estructura-del-proyecto)
7. [Configuración](#configuración)
8. [Archivos que genera la aplicación](#archivos-que-genera-la-aplicación)
9. [Limitaciones conocidas](#limitaciones-conocidas)
10. [Solución de problemas](#solución-de-problemas)

---

## Requisitos

- Windows, macOS o Linux con **cámara web**.
- **Python 3** en una versión compatible con MediaPipe y PyTorch.
- No se necesita GPU: el entrenamiento y la predicción corren en CPU.

Dependencias (ver `requirements.txt`): `pyside6`, `opencv-python`, `mediapipe`,
`numpy`, `torch` y `cryptography`.

## Instalación

```bash
git clone https://github.com/JavierSantiagoVera/TrackHandsGestures.git
cd TrackHandsGestures

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

El modelo de detección de manos de MediaPipe (`hand_landmarker.task`) ya viene
incluido en el repositorio.

## Uso

```bash
python run.py
```

### 1. Pantalla de inicio (consentimiento y contraseña)

- Lee la información sobre qué datos se recopilan y marca la casilla de aceptación.
- **Primera vez:** elige una contraseña para el grupo (entre 4 y 32 caracteres)
  y confírmala. Esta contraseña es **solo para el uso de ese grupo o persona**.
  No se comparte con nadie más.
- **Sesiones siguientes:** si ya hay datos guardados, la aplicación pide la misma
  contraseña. Sin ella no se puede entrar.
- **Borrar todo:** elimina el dataset, la contraseña y el modelo entrenado. Úsalo
  si olvidaste la contraseña o quieres empezar de cero. **No se puede deshacer.**

### 2. Ventana principal

| Zona | Para qué sirve |
|---|---|
| Video (izquierda) | Imagen de la cámara con los puntos de la mano dibujados. El borde muestra el progreso de la grabación. |
| Landmarks activos | Dibujo de la mano: haz clic en un punto para activarlo o desactivarlo. Los puntos desactivados no se usan como información para el modelo. |
| Predicción | El gesto reconocido y su nivel de confianza. |
| Clases | Nombre e icono de cada uno de los 3 gestos. El icono aparece sobre el video cuando se reconoce ese gesto. |
| ⏺ Grabar gesto | Tras una cuenta regresiva de 3 s, graba una muestra de 60 fotogramas del gesto. |
| ↶ (junto a cada gesto) | Borra **solo la última muestra** de ese gesto. |
| Cancelar grabación | Detiene la cuenta regresiva o la grabación en curso. |
| Resetear dataset | Borra todas las muestras (la contraseña se conserva). |
| ⚡ Entrenar modelo | Entrena el clasificador con las muestras grabadas. Siempre visible al pie del panel. |

### Flujo recomendado

1. Pon nombre a los 3 gestos (opcional: elige un icono para cada uno).
2. Graba **varias muestras de cada gesto** (se recomiendan al menos 10–15 por
   gesto), variando un poco la posición y la distancia a la cámara.
3. Pulsa **Entrenar modelo** y espera el mensaje de confirmación.
4. Haz los gestos frente a la cámara: la predicción aparece en el panel derecho.
5. Si un gesto se confunde, graba más muestras de ese gesto y vuelve a entrenar.

Cuando el dataset queda vacío, los nombres de los gestos vuelven a los valores
por defecto (`gesto 1`, `gesto 2`, `gesto 3`).

## Privacidad y seguridad

**Qué se guarda:** secuencias de características numéricas calculadas a partir de
los 21 puntos de la mano (distancias y ángulos). **No se guarda video, imágenes
ni audio.**

**Cifrado del dataset (`dataset_lsc_3gestos.enc`):**

- La clave se deriva de la contraseña con **PBKDF2-HMAC-SHA256**
  (390 000 iteraciones y una sal aleatoria de 16 bytes por archivo).
- Los datos se cifran con **Fernet** (AES-128-CBC + HMAC-SHA256) de la librería
  `cryptography`. Fernet es autenticado: con una contraseña incorrecta el
  descifrado falla; nunca devuelve datos corruptos.
- El archivo se escribe de forma atómica (primero a un temporal y luego se
  reemplaza), para que un cierre inesperado no lo deje dañado.

**Verificación de la contraseña (`auth.enc`):**

- Al crear la sesión se guarda una frase conocida, cifrada con la contraseña.
- Al entrar, la aplicación intenta descifrarla; si el resultado coincide con la
  frase, la contraseña es correcta. Así se puede validar sin descifrar todo el
  dataset y la sesión queda protegida aunque todavía no haya muestras.
- La contraseña en sí **no se guarda en ningún lado**.

**Qué NO está cifrado:**

- `lstm_3gestures.pt` (el modelo entrenado). Contiene pesos de la red neuronal,
  no las muestras, pero se obtiene a partir de ellas. **Borrar todo** también lo
  elimina.
- `ui_classes.json` (nombres de los gestos y rutas de los iconos).

**Repositorio:** el `.gitignore` excluye los datasets (`*.enc`, `*.npz`), los
pesos entrenados (`*.pt`, `*.pth`, `*.ckpt`, `*.onnx`, `*.safetensors`) y el
verificador de contraseña, para que ningún dato de participantes llegue a git.

> **Si se olvida la contraseña, los datos no se pueden recuperar.** La única
> opción es usar **Borrar todo** y empezar una sesión nueva.

## Cómo funciona

```
Cámara ──► MediaPipe Hand Landmarker ──► 21 puntos (x, y, z)
                                             │
                                             ▼
                             Vector de 64 características por fotograma
                             (distancias, ángulos, forma de la mano)
                                             │
                                             ▼
                            Ventana de 60 fotogramas (≈ 2 s de gesto)
                                             │
                                             ▼
                        Red convolucional temporal (TemporalCNN, 1D)
                                             │
                                             ▼
                          Probabilidad de cada gesto ──► predicción
```

1. **Detección** (`app/worker.py`): un hilo lee la cámara y usa el modelo
   *Hand Landmarker* de MediaPipe para obtener los 21 puntos de la mano.
2. **Características** (`app/features.py`): por cada fotograma se calcula un
   vector de tamaño fijo (64) con:
   - 21 longitudes de las conexiones de la mano, normalizadas por su tamaño;
   - 15 ángulos de las articulaciones de los dedos;
   - 21 distancias de cada punto a la muñeca;
   - relleno con ceros hasta 64.

   Los puntos desactivados en "Landmarks activos" aportan 0, así el modelo
   aprende sin esa información. Al normalizar por el tamaño de la mano, las
   características no dependen de la distancia a la cámara.
3. **Grabación**: una muestra son 60 vectores consecutivos (una matriz 60 × 64)
   con la etiqueta del gesto. Se añade al dataset cifrado de inmediato.
4. **Entrenamiento** (`app/train_worker.py`): una red `TemporalCNN`
   (`app/model.py`) con dos capas `Conv1d` sobre el eje del tiempo y un
   *pooling* global. Se entrena con AdamW y *early stopping*: se detiene si la
   pérdida no mejora durante varias épocas. Corre en segundo plano para no
   congelar la interfaz.
5. **Predicción en tiempo real** (`app/classifier.py`): un búfer circular guarda
   los últimos 60 vectores. Cada cierto número de fotogramas se evalúa el modelo.
   Solo se muestra el gesto si la confianza supera el umbral, y se mantiene unos
   segundos para que la etiqueta no parpadee.

## Estructura del proyecto

```
TrackHandsGestures/
├── run.py                  Punto de entrada: diálogo de inicio + ventana principal
├── requirements.txt        Dependencias de Python
├── hand_landmarker.task    Modelo de MediaPipe para detectar manos
├── ui_classes.json         Nombres e iconos de los gestos (se reescribe al usar la app)
└── app/
    ├── config.py           Parámetros globales (rutas, tamaños, entrenamiento)
    ├── consent_dialog.py   Consentimiento, creación y verificación de contraseña
    ├── crypto_store.py     Cifrado/descifrado del dataset y verificador de contraseña
    ├── dataset_store.py    Muestras en memoria + guardado cifrado + deshacer
    ├── main_window.py      Ventana principal (interfaz, grabación, entrenamiento)
    ├── worker.py           Hilo de cámara: detección, dibujo, grabación, predicción
    ├── features.py         Conversión de puntos de la mano a vector de características
    ├── model.py            Arquitectura de la red (TemporalCNN)
    ├── train_worker.py     Entrenamiento en segundo plano
    ├── classifier.py       Predicción en tiempo real con búfer y umbral
    ├── landmark_widget.py  Selector de puntos de la mano (clic para activar/desactivar)
    ├── rec_border.py       Borde del video que muestra el progreso de grabación
    ├── mp_draw.py          Dibujo de los puntos sobre el video
    ├── qt_utils.py         Conversión de imágenes OpenCV → Qt
    └── theme.py            Estilos (tema oscuro)
```

## Configuración

Los parámetros están en `app/config.py`:

| Parámetro | Valor | Descripción |
|---|---|---|
| `SEQ_LEN` | 60 | Fotogramas por muestra y tamaño de la ventana de predicción. |
| `FEATURE_DIM` | 64 | Tamaño del vector de características. Debe coincidir con `features.py`. |
| `DETECT_W` | 640 | Ancho al que se reduce la imagen antes de detectar (menos = más FPS). |
| `TRAIN_BATCH` | 32 | Tamaño de lote en el entrenamiento. |
| `TRAIN_EPOCHS` | 40 | Máximo de épocas de entrenamiento (el *early stopping* suele cortar antes). |
| `TRAIN_LR` / `TRAIN_WD` | 1e-3 / 1e-4 | Tasa de aprendizaje y *weight decay* (AdamW). |
| `TRAIN_PATIENCE` | 6 | Épocas sin mejora antes de detener el entrenamiento. |
| `PREDICT_EVERY` | 15 | Cada cuántos fotogramas se evalúa el modelo. |
| `CONF_THRESH` | 0.75 | Confianza mínima para mostrar una predicción. |
| `HOLD_SECONDS` | 2.0 | Tiempo que se mantiene la última predicción válida. |

> Si cambias `SEQ_LEN` o `FEATURE_DIM`, los datasets y modelos anteriores dejan
> de ser compatibles: usa **Borrar todo** y graba de nuevo.

## Archivos que genera la aplicación

Todos se crean en la carpeta desde la que se ejecuta `run.py` y ninguno se sube
a git:

| Archivo | Contenido | Cifrado |
|---|---|---|
| `dataset_lsc_3gestos.enc` | Muestras grabadas | Sí |
| `auth.enc` | Verificador de la contraseña | Sí |
| `lstm_3gestures.pt` | Pesos del modelo entrenado | No |
| `ui_classes.json` | Nombres e iconos de los gestos | No |

## Limitaciones conocidas

- El número de gestos está fijo en **3**.
- Las características se calculan con **una sola mano** (la primera que detecta
  MediaPipe), aunque en el video se dibujen dos.
- Un solo dataset por carpeta: para que varios grupos usen el mismo computador,
  cada uno debe ejecutar la aplicación desde una copia distinta de la carpeta.
- El modelo es pequeño y está pensado para pocos gestos y pocas muestras; no es
  un traductor general de LSC.

## Solución de problemas

| Problema | Qué hacer |
|---|---|
| "Contraseña incorrecta" | Verifica mayúsculas y espacios. Si no la recuerdas, usa **Borrar todo**. |
| La cámara no abre | Cierra otras aplicaciones que la usen y revisa los permisos de cámara del sistema. |
| La interfaz se ve cortada con escala de pantalla al 125 % / 150 % | El panel derecho tiene scroll y el botón de entrenar siempre queda visible abajo. Maximiza la ventana. |
| Las predicciones son malas | Graba más muestras por gesto, con variedad, y vuelve a entrenar. Revisa que los puntos importantes estén activos. |
| "Dataset no cargado" | El archivo está dañado o se cambió `SEQ_LEN`/`FEATURE_DIM`. Usa **Borrar todo**. |
