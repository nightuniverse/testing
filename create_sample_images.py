#!/usr/bin/env python3
"""
샘플 이미지 생성 스크립트
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_sample_images():
    """샘플 이미지들 생성"""
    
    # 폴더 생성
    os.makedirs("rag_anything_output/품질검사표", exist_ok=True)
    os.makedirs("rag_anything_output/조립공정도", exist_ok=True)
    os.makedirs("rag_anything_output/부품도면", exist_ok=True)
    
    # 1. 품질검사표 이미지 생성
    img1 = Image.new('RGB', (800, 600), color='white')
    draw1 = ImageDraw.Draw(img1)
    
    # 제목
    draw1.text((400, 50), "품질 검사표", fill='black', anchor='mm')
    
    # 검사 항목들
    items = [
        "1. 부품 외관 검사",
        "2. 치수 정밀도 검사", 
        "3. 기능 테스트",
        "4. 품질 등급 분류",
        "5. 최종 승인"
    ]
    
    for i, item in enumerate(items):
        y = 120 + i * 80
        draw1.text((100, y), item, fill='blue', font=None)
        draw1.rectangle([(80, y-10), (90, y+10)], fill='green')
    
    # 기준 정보
    draw1.text((400, 500), "검사 기준: A급 이상 (98% 합격률)", fill='red', anchor='mm')
    
    # 테이블 형태로 추가 정보
    draw1.rectangle([(50, 200), (750, 450)], fill='lightgray', outline='black', width=2)
    draw1.text((100, 220), "검사 항목", fill='black')
    draw1.text((300, 220), "기준값", fill='black')
    draw1.text((500, 220), "측정값", fill='black')
    draw1.text((650, 220), "합격여부", fill='black')
    
    # 구분선
    draw1.line([(50, 250), (750, 250)], fill='black', width=2)
    
    # 검사 데이터
    test_data = [
        ("외관", "깨끗함", "깨끗함", "합격"),
        ("치수", "±0.1mm", "0.05mm", "합격"),
        ("기능", "정상작동", "정상작동", "합격"),
        ("강도", "≥100MPa", "120MPa", "합격")
    ]
    
    for i, (item, standard, measured, result) in enumerate(test_data):
        y = 280 + i * 40
        draw1.text((100, y), item, fill='black')
        draw1.text((300, y), standard, fill='blue')
        draw1.text((500, y), measured, fill='green')
        draw1.text((650, y), result, fill='green')
    
    img1.save("rag_anything_output/품질검사표/품질검사표.png", quality=95)
    
    # 2. 조립공정도 이미지 생성
    img2 = Image.new('RGB', (800, 600), color='lightblue')
    draw2 = ImageDraw.Draw(img2)
    
    # 제목
    draw2.text((400, 50), "조립 공정도", fill='darkblue', anchor='mm')
    
    # 공정 단계들
    processes = [
        "수입검사 (30분)",
        "전처리 (45분)", 
        "조립 (120분)",
        "검사 (60분)",
        "포장 (30분)"
    ]
    
    for i, process in enumerate(processes):
        x = 150 + i * 120
        y = 300
        draw2.ellipse([(x-40, y-40), (x+40, y+40)], fill='white', outline='blue', width=3)
        draw2.text((x, y), str(i+1), fill='blue', anchor='mm')
        draw2.text((x, y+80), process, fill='darkblue', anchor='mm')
        
        # 화살표 그리기
        if i < len(processes) - 1:
            draw2.line([(x+40, y), (x+80, y)], fill='red', width=3)
    
    img2.save("rag_anything_output/조립공정도/조립공정도.png", quality=95)
    
    # 3. 부품도면 이미지 생성
    img3 = Image.new('RGB', (800, 600), color='lightgray')
    draw3 = ImageDraw.Draw(img3)
    
    # 제목
    draw3.text((400, 50), "부품 도면", fill='black', anchor='mm')
    
    # 도면 요소들
    # 외곽선
    draw3.rectangle([(200, 150), (600, 450)], fill='white', outline='black', width=2)
    
    # 구멍들
    draw3.ellipse([(250, 200), (270, 220)], fill='white', outline='red', width=2)
    draw3.ellipse([(530, 200), (550, 220)], fill='white', outline='red', width=2)
    draw3.ellipse([(250, 380), (270, 400)], fill='white', outline='red', width=2)
    draw3.ellipse([(530, 380), (550, 400)], fill='white', outline='red', width=2)
    
    # 치수선
    draw3.line([(200, 500), (600, 500)], fill='blue', width=1)
    draw3.text((400, 520), "400mm", fill='blue', anchor='mm')
    
    draw3.line([(650, 150), (650, 450)], fill='blue', width=1)
    draw3.text((670, 300), "300mm", fill='blue', anchor='mm')
    
    # 재질 정보
    draw3.text((400, 550), "재질: 알루미늄 합금, 공차: ±0.1mm", fill='darkgreen', anchor='mm')
    
    img3.save("rag_anything_output/부품도면/부품도면.png", quality=95)
    
    print("✅ 샘플 이미지 생성 완료!")
    print("• 품질검사표.png")
    print("• 조립공정도.png") 
    print("• 부품도면.png")

if __name__ == "__main__":
    create_sample_images()
