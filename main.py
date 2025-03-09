import streamlit as st
from PIL import Image
import google.generativeai as genai
import io
import base64


# configurar a chave da api
chave = st.secrets["GEMINI_CHAVE"]
genai.configure(api_key=chave)

model = genai.GenerativeModel("gemini-1.5-flash")


# função para analise de atributos
def analyze_human_attributes(image):
    prompt = """
    Você é uma IA treinada para analisar atributos humanos a partir de imagens com alta precisão.
    Analise cuidadosamente a imagem fornecida e retorne os seguintes detalhes:
    - **Gênero** (Masculino/Feminino)
    - **Estimativa de Idade** (ex: 25 anos)
    - **Etnia** (ex: Asiático, Caucasiano, Africano, etc.)
    - **Humor** (ex: Feliz, Triste, Neutro, Animado)
    - **Expressão Facial** (ex: Sorrindo, Franzindo a testa, Neutro, etc.)
    - **Óculos** (Sim/Não)
    - **Barba** (Sim/Não)
    - **Cor do Cabelo** (ex: Preto, Loiro, Castanho)
    - **Cor dos Olhos** (ex: Azul, Verde, Castanho)
    - **Acessório na Cabeça** (Sim/Não, especifique o tipo se aplicável)
    - **Emoções Detectadas** (ex: Alegre, Concentrado, Bravo, etc.)
    - **Nível de Confiança** (Precisão da previsão em porcentagem)
    """

    # converter imagem para bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)  # Reposicionar o ponteiro para o início do BytesIO

    # criar conteúdo para envio ao modelo
    # esta função envia a solicitação para o modelo gemini
    response = model.generate_content(
        [
            prompt,
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_bytes.getvalue()).decode('utf-8')
            }
        ]
    )

    return response.text.strip()


# cria a interface no streamlit
st.title("Detecção de Atributos Humanos")
st.write("Carregue uma imagem para detectar atributos humanos com IA")

# fazer o upload da imagem
uploaded_image = st.file_uploader("Carregue uma imagem", type=['png', 'jpg', 'jpeg'])

if uploaded_image:
    img = Image.open(uploaded_image)

    # exibe a imagem e o resultado um ao lado do outro
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption='Imagem Carregada', use_container_width=True)

    with col2:
        with st.spinner("Analisando imagem..."):
            person_info = analyze_human_attributes(img)
            st.markdown(person_info)
