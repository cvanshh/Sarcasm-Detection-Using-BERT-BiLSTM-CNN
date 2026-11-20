"""Sarcasm detector UI (BERT + BiLSTM + CNN).

Run from the folder that contains the files saved by the notebook:
    sarcasm_bert_pytorch.pth
    sarcasm_bert_tokenizer/

    pip install streamlit torch transformers
    streamlit run app.py
"""

import html
import re
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

MODEL_PATH = Path("sarcasm_bert_pytorch.pth")
TOKENIZER_DIR = Path("sarcasm_bert_tokenizer")
THRESHOLD = 0.5

st.set_page_config(page_title="Sarcasm detector", page_icon="🎭", layout="centered")


# ───────────────────────── Model (same as the notebook) ─────────────────────────
class HierarchicalBERT(nn.Module):
    def __init__(
        self, bert_model, lstm_units=128, cnn_filters=128, dense_units=128, dropout=0.3
    ):
        super().__init__()
        self.bert = bert_model
        hidden = self.bert.config.hidden_size
        self.dense_sentence = nn.Linear(hidden, hidden)
        self.bilstm_encoder = nn.LSTM(
            hidden, lstm_units, batch_first=True, bidirectional=True
        )
        self.conv = nn.Conv1d(lstm_units * 2, cnn_filters, kernel_size=3, padding=1)
        self.dense_df = nn.Linear(cnn_filters, dense_units)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(dense_units, 1)

    def forward(self, input_ids, attention_mask):
        x = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        x = torch.relu(self.dense_sentence(x))
        x, _ = self.bilstm_encoder(x)
        x = torch.relu(self.conv(x.transpose(1, 2)))
        mask = attention_mask.unsqueeze(1).to(x.dtype)
        x = (x * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)
        x = self.dropout(torch.relu(self.dense_df(x)))
        return self.output_layer(x).squeeze(1)


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    tokenizer = BertTokenizer.from_pretrained(str(TOKENIZER_DIR))
    model = HierarchicalBERT(
        BertModel.from_pretrained(ckpt["model_name"]),
        lstm_units=ckpt["lstm_units"],
        cnn_filters=ckpt["cnn_filters"],
        dense_units=ckpt["dense_units"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), tokenizer, ckpt["max_length"], device


def predict(text):
    model, tokenizer, max_length, device = load_model()
    cleaned = re.sub(r"[^a-zA-Z\s]", "", str(text)).lower().strip()
    enc = tokenizer(
        cleaned,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        logits = model(enc["input_ids"], enc["attention_mask"])
    prob = torch.sigmoid(logits.float()).item()
    n = int(enc["attention_mask"].sum())
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0][:n])
    return prob, tokens


# ───────────────────────────────── Styling ─────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=DM+Sans:wght@400;500&display=swap');
@property --p { syntax: '<integer>'; initial-value: 0; inherits: false; }

:root{
  --bg:#150F24; --surface:#201836; --line:#382D57; --text:#F2EDFF; --muted:#9C8FBD;
  --sincere:#35D8C4; --sarcasm:#FF4F8B;
}
#MainMenu, header, footer, [data-testid="stToolbar"]{ display:none !important; }
.stApp{
  background:
    radial-gradient(900px 500px at 15% -10%, #2A1B52 0%, transparent 60%),
    radial-gradient(700px 420px at 100% 0%, #3A1640 0%, transparent 55%),
    var(--bg);
  color:var(--text); font-family:'DM Sans',sans-serif;
}
.block-container{ max-width:820px; padding-top:3.2rem; }

.title{ font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:3.1rem;
  line-height:1.02; letter-spacing:-.03em; margin:0 0 .6rem;
  animation: rise .9s cubic-bezier(.2,.8,.2,1) both; }
.title span{ display:inline-block; animation: tilt 5s ease-in-out 1s infinite; transform-origin:50% 100%; }
.sub{ color:var(--muted); font-size:1.05rem; max-width:34rem; margin-bottom:1.6rem;
  animation: rise .9s .12s cubic-bezier(.2,.8,.2,1) both; }
@keyframes rise{ from{opacity:0; transform:translateY(18px)} to{opacity:1; transform:none} }
@keyframes tilt{ 0%,86%,100%{transform:rotate(0)} 90%{transform:rotate(-5deg)} 95%{transform:rotate(4deg)} }

/* input */
[data-testid="stForm"]{ border:none; padding:0; }
.stTextArea textarea{
  background:var(--surface) !important; color:var(--text) !important;
  border:1.5px solid var(--line) !important; border-radius:18px !important;
  font-size:1.1rem !important; padding:1.1rem 1.2rem !important; line-height:1.5;
  transition:border-color .25s, box-shadow .25s, transform .25s;
}
.stTextArea textarea::placeholder{ color:#6E6190 !important; }
.stTextArea textarea:focus{
  border-color:var(--sarcasm) !important; transform:translateY(-2px);
  box-shadow:0 0 0 4px rgba(255,79,139,.16), 0 18px 40px -18px rgba(255,79,139,.55) !important;
}
.stTextArea label{ display:none; }
[data-testid="stFormSubmitButton"] button{
  position:relative; overflow:hidden; border:none; border-radius:14px; padding:.7rem 1.8rem;
  font-family:'Bricolage Grotesque',sans-serif; font-weight:600; font-size:1.05rem; color:#150F24;
  background:linear-gradient(100deg,var(--sincere),var(--sarcasm)); background-size:160% 100%;
  transition:background-position .5s, transform .2s, box-shadow .3s;
}
[data-testid="stFormSubmitButton"] button:hover{
  background-position:100% 0; transform:translateY(-2px);
  box-shadow:0 14px 30px -10px rgba(255,79,139,.6); color:#150F24;
}
[data-testid="stFormSubmitButton"] button:active{ transform:scale(.97); }
[data-testid="stFormSubmitButton"] button p{ color:#150F24 !important; }

/* gauge */
.stage{ margin:2rem 0 .5rem; text-align:center; }
.gauge{ width:min(100%,460px); overflow:visible; }
.arc{ stroke-dasharray:503; stroke-dashoffset:503; animation: draw 1.3s .1s cubic-bezier(.6,0,.2,1) forwards; }
@keyframes draw{ to{ stroke-dashoffset:0 } }
.needle{ transform-origin:200px 200px; transform:rotate(-90deg); }
.needle.go{ animation: sweep 1.7s 1.45s cubic-bezier(.3,1.5,.5,1) forwards; }
.needle.idle{ transform:rotate(0); animation: sway 4.5s ease-in-out infinite; }
@keyframes sweep{ to{ transform:rotate(var(--a)) } }
@keyframes sway{ 0%,100%{ transform:rotate(-7deg) } 50%{ transform:rotate(7deg) } }
.glow{ opacity:0; animation: fade .8s 2.7s forwards; }
@keyframes fade{ to{ opacity:1 } }

.readout{ margin-top:-.4rem; }
.num{ font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:4.6rem; line-height:1;
  letter-spacing:-.04em; counter-reset:p var(--p); animation: count 1.6s 1.5s cubic-bezier(.2,.7,.2,1) forwards; }
.num::after{ content:counter(p) "%"; }
@keyframes count{ to{ --p:var(--t) } }
.verdict{ font-family:'Bricolage Grotesque',sans-serif; font-weight:600; font-size:1.5rem; margin-top:.3rem;
  opacity:0; transform:translateY(10px); animation: pop .6s 2.9s cubic-bezier(.2,.9,.3,1.3) forwards; }
@keyframes pop{ to{ opacity:1; transform:none } }
.hint{ color:var(--muted); font-size:.95rem; margin-top:.4rem; }
.sar{ color:var(--sarcasm); } .sin{ color:var(--sincere); }

/* pipeline */
.pipe{ display:flex; align-items:center; margin:2.4rem 0 1.4rem; }
.node{ flex:0 0 auto; padding:.45rem .95rem; border-radius:999px; border:1.5px solid var(--line);
  color:var(--muted); font-size:.9rem; font-weight:500; background:var(--surface);
  animation: lit .45s forwards; animation-delay:calc(var(--i) * .3s + .2s); }
@keyframes lit{ to{ color:var(--text); border-color:var(--c); box-shadow:0 0 22px -4px var(--c); } }
.link{ flex:1; height:2px; background:var(--line); position:relative; overflow:hidden; min-width:10px; }
.link::after{ content:""; position:absolute; inset:0; background:var(--c); transform:scaleX(0);
  transform-origin:left; animation: fill .3s forwards; animation-delay:calc(var(--i) * .3s + .45s); }
@keyframes fill{ to{ transform:scaleX(1) } }

/* tokens */
.tokens{ display:flex; flex-wrap:wrap; gap:.45rem; }
.tok{ padding:.28rem .65rem; border-radius:8px; background:var(--surface); border:1px solid var(--line);
  font-size:.92rem; opacity:0; transform:translateY(8px) scale(.9);
  animation: tokin .4s cubic-bezier(.2,.9,.3,1.3) forwards; animation-delay:calc(var(--i) * .045s + .1s); }
.tok.sp{ color:var(--muted); border-style:dashed; }
@keyframes tokin{ to{ opacity:1; transform:none } }

.err{ border:1.5px solid var(--sarcasm); border-radius:14px; padding:1rem 1.2rem; background:rgba(255,79,139,.08); }

@media (prefers-reduced-motion: reduce){
  *{ animation-duration:.01s !important; animation-delay:0s !important; }
}
@media (max-width:600px){ .title{ font-size:2.3rem } .num{ font-size:3.6rem } .pipe{ font-size:.8rem } .node{ padding:.35rem .6rem } }
</style>
"""

GAUGE = """
<svg class="gauge" viewBox="0 0 400 230" role="img" aria-label="Sarcasm gauge">
  <defs>
    <linearGradient id="g" x1="0" x2="1">
      <stop offset="0" stop-color="#35D8C4"/><stop offset=".5" stop-color="#8B7BFF"/><stop offset="1" stop-color="#FF4F8B"/>
    </linearGradient>
    <filter id="b"><feGaussianBlur stdDeviation="9"/></filter>
  </defs>
  <path d="M40 200 A160 160 0 0 1 360 200" fill="none" stroke="#2C2247" stroke-width="22" stroke-linecap="round"/>
  <path class="glow" d="M40 200 A160 160 0 0 1 360 200" fill="none" stroke="url(#g)" stroke-width="22"
        stroke-linecap="round" filter="url(#b)" opacity=".0" style="--x:0"/>
  <path class="arc" d="M40 200 A160 160 0 0 1 360 200" fill="none" stroke="url(#g)" stroke-width="22" stroke-linecap="round"/>
  <line x1="200" y1="22" x2="200" y2="50" stroke="#F2EDFF" stroke-opacity=".5" stroke-width="2"/>
  <g class="needle {cls}" style="--a:{angle}deg">
    <path d="M200 52 L208 200 L192 200 Z" fill="#F2EDFF"/>
  </g>
  <circle cx="200" cy="200" r="13" fill="#F2EDFF"/><circle cx="200" cy="200" r="5" fill="#150F24"/>
  <text x="40" y="226" fill="#9C8FBD" font-size="14" text-anchor="middle" font-family="DM Sans">Sincere</text>
  <text x="360" y="226" fill="#9C8FBD" font-size="14" text-anchor="middle" font-family="DM Sans">Sarcastic</text>
</svg>
"""

STEPS = ["Tokenize", "BERT", "BiLSTM", "CNN", "Verdict"]


def pipeline(color):
    parts = []
    for i, name in enumerate(STEPS):
        parts.append(f'<div class="node" style="--i:{i};--c:{color}">{name}</div>')
        if i < len(STEPS) - 1:
            parts.append(f'<div class="link" style="--i:{i};--c:{color}"></div>')
    return f'<div class="pipe">{"".join(parts)}</div>'


def tokens_html(tokens, limit=48):
    chips = []
    for i, t in enumerate(tokens[:limit]):
        cls = "tok sp" if t in ("[CLS]", "[SEP]") else "tok"
        chips.append(f'<span class="{cls}" style="--i:{i}">{html.escape(t)}</span>')
    if len(tokens) > limit:
        chips.append(
            f'<span class="tok sp" style="--i:{limit}">+{len(tokens) - limit} more</span>'
        )
    return f'<div class="tokens">{"".join(chips)}</div>'


def render_idle():
    st.markdown(
        f'<div class="stage">{GAUGE.replace("{cls}", "idle").replace("{angle}", "0")}'
        '<div class="hint">Write a comment and the needle will settle on a score.</div></div>',
        unsafe_allow_html=True,
    )


def render_result(prob, tokens):
    sarcastic = prob >= THRESHOLD
    color = "#FF4F8B" if sarcastic else "#35D8C4"
    angle = round((prob - 0.5) * 180, 1)
    pct = round(prob * 100)
    verdict = "Sarcastic" if sarcastic else "Not sarcastic"
    cls = "sar" if sarcastic else "sin"
    gauge = GAUGE.replace("{cls}", "go").replace("{angle}", str(angle))
    st.markdown(
        f'<div class="stage">{gauge}<div class="readout">'
        f'<div class="num {cls}" style="--t:{pct}"></div>'
        f'<div class="verdict {cls}">{verdict}</div></div></div>'
        f"{pipeline(color)}{tokens_html(tokens)}",
        unsafe_allow_html=True,
    )


# ───────────────────────────────── Page ────────────────────────────────────────
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="title">Is that comment<br>being <span>sarcastic?</span></div>'
    '<div class="sub">Paste a comment and a BERT + BiLSTM + CNN model scores how likely it is to be sarcasm.</div>',
    unsafe_allow_html=True,
)

with st.form("f", border=False):
    text = st.text_area(
        "Comment",
        placeholder="Oh great, another meeting. Just what I needed.",
        height=130,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Analyze comment")

if not (MODEL_PATH.exists() and TOKENIZER_DIR.exists()):
    st.markdown(
        f'<div class="err">Model files not found. Run the notebook’s save cell, then start this app '
        f"from the same folder so it can see <b>{MODEL_PATH}</b> and <b>{TOKENIZER_DIR}/</b>.</div>",
        unsafe_allow_html=True,
    )
elif submitted and text.strip():
    prob, tokens = predict(text)
    render_result(prob, tokens)
elif submitted:
    st.markdown('<div class="err">Enter a comment first.</div>', unsafe_allow_html=True)
else:
    render_idle()
