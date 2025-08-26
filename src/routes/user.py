from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from datetime import datetime

from src.models.user import User, db

user_bp = Blueprint('user', __name__)

# Configurações
UPLOAD_FOLDER = 'uploads/profiles'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_upload_folder():
    """Garante que a pasta de uploads existe"""
    upload_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), UPLOAD_FOLDER)
    if not os.path.exists(upload_path):
        os.makedirs(upload_path)
    return upload_path

def extract_face_features(image_path):
    """Extrai características da face para armazenamento"""
    try:
        # Carrega o classificador de faces do OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Lê a imagem
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Converte para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detecta faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
        
        # Pega a primeira face detectada
        x, y, w, h = faces[0]
        face_roi = image[y:y+h, x:x+w]
        
        # Redimensiona para tamanho padrão
        face_resized = cv2.resize(face_roi, (100, 100))
        
        # Converte para escala de cinza e normaliza
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        face_normalized = face_gray.flatten().astype(np.float32) / 255.0
        
        return face_normalized
    except Exception as e:
        print(f"Erro na extração de características: {e}")
        return None

@user_bp.route('/users', methods=['GET'])
def get_users():
    users = User.query.all()
    return jsonify([user.to_dict() for user in users])

@user_bp.route('/users', methods=['POST'])
def create_user():
    try:
        # Verifica se é multipart/form-data (com arquivo)
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Dados do formulário
            username = request.form.get('username')
            email = request.form.get('email')
            phone = request.form.get('phone')
            
            if not username or not email:
                return jsonify({'error': 'username e email são obrigatórios'}), 400
            
            # Verifica se usuário já existe
            if User.query.filter_by(username=username).first():
                return jsonify({'error': 'Username já existe'}), 400
            if User.query.filter_by(email=email).first():
                return jsonify({'error': 'Email já existe'}), 400
            
            user = User(username=username, email=email, phone=phone)
            
            # Processa imagem de perfil se fornecida
            if 'profile_image' in request.files:
                file = request.files['profile_image']
                if file and file.filename != '' and allowed_file(file.filename):
                    upload_path = ensure_upload_folder()
                    filename = secure_filename(f"{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
                    file_path = os.path.join(upload_path, filename)
                    file.save(file_path)
                    
                    # Extrai características faciais
                    face_encoding = extract_face_features(file_path)
                    if face_encoding is not None:
                        user.set_face_encoding(face_encoding)
                        user.profile_image_path = file_path
                    else:
                        return jsonify({'error': 'Não foi possível detectar face na imagem fornecida'}), 400
            
            db.session.add(user)
            db.session.commit()
            
            return jsonify({
                'message': 'Usuário criado com sucesso',
                'user': user.to_dict()
            }), 201
        
        else:
            # Dados JSON (sem imagem)
            data = request.json
            if not data or 'username' not in data or 'email' not in data:
                return jsonify({'error': 'username e email são obrigatórios'}), 400
            
            # Verifica se usuário já existe
            if User.query.filter_by(username=data['username']).first():
                return jsonify({'error': 'Username já existe'}), 400
            if User.query.filter_by(email=data['email']).first():
                return jsonify({'error': 'Email já existe'}), 400
            
            user = User(
                username=data['username'], 
                email=data['email'],
                phone=data.get('phone')
            )
            db.session.add(user)
            db.session.commit()
            
            return jsonify({
                'message': 'Usuário criado com sucesso (sem reconhecimento facial)',
                'user': user.to_dict()
            }), 201
            
    except Exception as e:
        return jsonify({'error': f'Erro ao criar usuário: {str(e)}'}), 500

@user_bp.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = User.query.get_or_404(user_id)
    return jsonify(user.to_dict())

@user_bp.route('/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    try:
        user = User.query.get_or_404(user_id)
        
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Atualização com possível nova imagem
            if 'username' in request.form:
                user.username = request.form['username']
            if 'email' in request.form:
                user.email = request.form['email']
            if 'phone' in request.form:
                user.phone = request.form['phone']
            
            # Processa nova imagem de perfil se fornecida
            if 'profile_image' in request.files:
                file = request.files['profile_image']
                if file and file.filename != '' and allowed_file(file.filename):
                    upload_path = ensure_upload_folder()
                    filename = secure_filename(f"{user.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
                    file_path = os.path.join(upload_path, filename)
                    file.save(file_path)
                    
                    # Extrai novas características faciais
                    face_encoding = extract_face_features(file_path)
                    if face_encoding is not None:
                        user.set_face_encoding(face_encoding)
                        user.profile_image_path = file_path
                    else:
                        return jsonify({'error': 'Não foi possível detectar face na nova imagem'}), 400
        else:
            # Atualização apenas de dados
            data = request.json
            user.username = data.get('username', user.username)
            user.email = data.get('email', user.email)
            user.phone = data.get('phone', user.phone)
            user.is_active = data.get('is_active', user.is_active)
        
        db.session.commit()
        return jsonify({
            'message': 'Usuário atualizado com sucesso',
            'user': user.to_dict()
        })
        
    except Exception as e:
        return jsonify({'error': f'Erro ao atualizar usuário: {str(e)}'}), 500

@user_bp.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        user = User.query.get_or_404(user_id)
        
        # Remove arquivo de imagem se existir
        if user.profile_image_path and os.path.exists(user.profile_image_path):
            os.remove(user.profile_image_path)
        
        db.session.delete(user)
        db.session.commit()
        
        return jsonify({'message': 'Usuário deletado com sucesso'})
    except Exception as e:
        return jsonify({'error': f'Erro ao deletar usuário: {str(e)}'}), 500

@user_bp.route('/users/<int:user_id>/profile-image', methods=['POST'])
def upload_profile_image(user_id):
    """Endpoint específico para upload de imagem de perfil"""
    try:
        user = User.query.get_or_404(user_id)
        
        if 'image' not in request.files:
            return jsonify({'error': 'Nenhuma imagem enviada'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
        
        if file and allowed_file(file.filename):
            upload_path = ensure_upload_folder()
            filename = secure_filename(f"{user.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
            file_path = os.path.join(upload_path, filename)
            file.save(file_path)
            
            # Extrai características faciais
            face_encoding = extract_face_features(file_path)
            if face_encoding is not None:
                user.set_face_encoding(face_encoding)
                user.profile_image_path = file_path
                db.session.commit()
                
                return jsonify({
                    'message': 'Imagem de perfil atualizada com sucesso',
                    'user': user.to_dict()
                })
            else:
                # Remove arquivo se não conseguiu detectar face
                os.remove(file_path)
                return jsonify({'error': 'Não foi possível detectar face na imagem'}), 400
        
        return jsonify({'error': 'Tipo de arquivo não permitido'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Erro ao fazer upload da imagem: {str(e)}'}), 500

@user_bp.route('/users/search', methods=['GET'])
def search_users():
    """Busca usuários por nome ou email"""
    try:
        query = request.args.get('q', '').strip()
        if not query:
            return jsonify({'users': []})
        
        users = User.query.filter(
            (User.username.contains(query)) | 
            (User.email.contains(query))
        ).limit(10).all()
        
        return jsonify({
            'users': [user.to_dict() for user in users],
            'query': query
        })
        
    except Exception as e:
        return jsonify({'error': f'Erro na busca: {str(e)}'}), 500
