from flask import Flask, render_template, request, Response, redirect, url_for, session, stream_with_context
from database import SessionLocal, User, Conversation
from werkzeug.security import generate_password_hash, check_password_hash
from MedBot import MedBot
from pathlib import Path
import mimetypes
import markdown as md
import html
import json

db = SessionLocal()
bot = MedBot()
app = Flask(__name__)
app.secret_key = "une-cle-secrete-pour-ta-session"

def auto_title(text: str) -> str:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch, gc
    m_id = "meta-llama/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(m_id)
    model = AutoModelForCausalLM.from_pretrained(
        m_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    prompt = f"### Instruction:\nDonne un titre concis (≤ 4 mots)\n\n{text}\n\n### Réponse:"
    out = model.generate(**tok(prompt, return_tensors="pt").to(model.device),
                         max_new_tokens=12, temperature=0.2)
    full = tok.decode(out[0], skip_special_tokens=True)
    lines = [l for l in full.split('\n') if l.strip()]
    if len(lines) > 1:
        title = lines[-1]
    else:
        title = lines[0]
    title = title.strip()[:60]
    del model; gc.collect(); torch.cuda.empty_cache()
    return title

@app.route("/", methods=["GET"])
def home():
    if "user_id" not in session:
        return redirect(url_for("login"))
    user = db.query(User).filter_by(id=session["user_id"]).first()
    if user is None:
        session.clear()
        return redirect(url_for("login"))
    convs = (db.query(Conversation)
                .filter_by(user_id=user.id)
                .filter(Conversation.messages != "[]")
                .order_by(Conversation.created_at.desc())
                .all())
    conv = None
    if "current_conversation" in session:
        conv = db.query(Conversation).filter_by(
            id=session["current_conversation"], user_id=user.id
        ).first()
    if conv is None and convs:
        conv = convs[0]

    history_html = ""
    if conv and conv.messages:
        for msg in json.loads(conv.messages):
            if msg["role"] != "system":
                history_html += format_bubble(msg["role"], msg["content"])


    return render_template(
        "index.html",
        history=history_html,
        conversations=convs,
        active_conv_id=conv.id if conv else None
    )

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = db.query(User).filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session["user_id"] = user.id
            return redirect(url_for("home"))
        return "❌ Identifiants invalides"
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if db.query(User).filter_by(username=username).first():
            return "❌ Cet utilisateur existe déjà."
        user = User(username=username, password_hash=generate_password_hash(password))
        db.add(user)
        db.commit()
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

def generate_stream(user_msg, file, file_path, history, conv, bot):
    bot.set_history(history)
    # Génère la réponse complète d'un coup
    reply = bot.chat(user_msg, image=file_path if file else None)
    # Affiche la bulle user
    yield format_bubble("user", [{"type": "text", "text": user_msg}])
    # Affiche la bulle assistant
    yield format_bubble("assistant", [{"type": "text", "text": reply}])
    # Mets à jour l'historique DB
    conv.messages = json.dumps(bot.messages)
    db.commit()


@app.route("/chat", methods=["POST"])
def chat():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user = db.query(User).filter_by(id=session["user_id"]).first()
    file = request.files.get("file")
    user_msg = request.form.get("message")

    conv_id = session.get("current_conversation")
    if conv_id:
        conv = db.query(Conversation).filter_by(id=conv_id, user_id=user.id).first()
    else:
        conv = db.query(Conversation).filter_by(user_id=user.id).order_by(Conversation.id.desc()).first()
    if not conv:
        conv = Conversation(user_id=user.id, title="Nouvelle conversation", messages="[]")
        db.add(conv)
        db.commit()
        session["current_conversation"] = conv.id

    file_path = None
    if file and file.filename:
        file_path = Path(f"uploads/{file.filename}")
        file.save(file_path)

    history = json.loads(conv.messages or "[]")

    # Ajoute le message système si l'historique est vide
    if len(history) == 0:
        with open("system_prompt.md", "r", encoding="utf-8") as f:
            system_prompt = f.read()
        history.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

   # history.append({"role": "user", "content": [{"type": "text", "text": user_msg}]})


    return Response(
        stream_with_context(generate_stream(user_msg, file, file_path, history, conv, bot)),
        mimetype="text/html"
    )

@app.route("/conversations")
def conversations():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user = db.query(User).filter_by(id=session["user_id"]).first()
    convs = db.query(Conversation).filter_by(user_id=user.id).order_by(Conversation.created_at.desc()).all()
    return render_template("conversations.html", conversations=convs)

@app.route("/new")
def new_conversation():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user = db.query(User).filter_by(id=session["user_id"]).first()
    conv = Conversation(user_id=user.id, title="Nouvelle conversation", messages="[]")
    db.add(conv)
    db.commit()
    session["current_conversation"] = conv.id
    return redirect(url_for("home"))

@app.route("/switch/<int:conv_id>")
def switch(conv_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    session["current_conversation"] = conv_id
    return redirect(url_for("home"))

def format_bubble(role, content):
    text = ""
    for part in content:
        if part.get("type") == "text":
            text += part.get("text", "")
        elif part.get("type") == "image":
            text += "<em>[Image]</em>"
    html_content = md.markdown(html.escape(text))
    return f'<div class="message {role}"><div class="bubble">{html_content}</div></div>'


if __name__ == "__main__":
    import os
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=False, port=5000, use_reloader=False)
