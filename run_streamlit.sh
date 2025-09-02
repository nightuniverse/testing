#!/bin/bash

echo "🚀 Test Excels VLM System - Streamlit Interface"
echo "================================================"

# Qdrant 서버 상태 확인
echo "🔍 Qdrant 서버 상태 확인 중..."
if curl -s http://localhost:6333 > /dev/null; then
    echo "✅ Qdrant 서버가 실행 중입니다."
else
    echo "❌ Qdrant 서버가 실행되지 않았습니다."
    echo "💡 다음 명령어로 Qdrant를 실행하세요:"
    echo "   docker run -d -p 6333:6333 --name qdrant-server qdrant/qdrant"
    echo ""
    read -p "Qdrant를 지금 실행하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🐳 Qdrant 서버를 시작하는 중..."
        docker run -d -p 6333:6333 --name qdrant-server qdrant/qdrant
        echo "⏳ Qdrant 서버가 시작될 때까지 잠시 기다리는 중..."
        sleep 10
    else
        echo "❌ Qdrant 서버 없이는 시스템을 실행할 수 없습니다."
        exit 1
    fi
fi

# Python 의존성 확인
echo "🐍 Python 의존성 확인 중..."
if ! python -c "import streamlit, pandas, qdrant_client" 2>/dev/null; then
    echo "❌ 필요한 Python 패키지가 설치되지 않았습니다."
    echo "💡 다음 명령어로 패키지를 설치하세요:"
    echo "   pip install -r requirements.txt"
    echo ""
    read -p "패키지를 지금 설치하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install -r requirements.txt
    else
        echo "❌ 필요한 패키지 없이는 시스템을 실행할 수 없습니다."
        exit 1
    fi
fi

echo "✅ 모든 의존성이 준비되었습니다."

# Streamlit 앱 실행
echo "🌐 Streamlit 앱을 시작하는 중..."
echo "🌍 브라우저에서 http://localhost:8501 로 접속하세요."
echo "🛑 종료하려면 Ctrl+C를 누르세요."
echo ""

python -m streamlit run streamlit_vlm_interface.py --server.port 8501
