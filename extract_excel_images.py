#!/usr/bin/env python3
"""
엑셀 파일에서 이미지 추출 스크립트
"""

import zipfile
import os
from pathlib import Path
from PIL import Image
import io

def extract_images_from_excel(excel_file_path):
    """엑셀 파일에서 이미지 추출"""
    excel_path = Path(excel_file_path)
    
    if not excel_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {excel_file_path}")
        return []
    
    print(f"🔍 엑셀 파일 분석 중: {excel_path.name}")
    
    # 엑셀 파일을 ZIP으로 열기 (엑셀은 ZIP 압축 파일입니다)
    try:
        with zipfile.ZipFile(excel_path, 'r') as zip_file:
            # 이미지 파일들 찾기
            image_files = []
            for file_info in zip_file.filelist:
                file_name = file_info.filename
                
                # 이미지 파일인지 확인
                if any(file_name.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']):
                    image_files.append(file_name)
            
            print(f"📸 발견된 이미지: {len(image_files)}개")
            
            if not image_files:
                print("   이미지가 포함되어 있지 않습니다.")
                return []
            
            # 이미지 추출 및 저장
            extracted_images = []
            output_dir = Path("rag_anything_output") / excel_path.stem / "extracted_images"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for image_file in image_files:
                try:
                    # 이미지 데이터 읽기
                    with zip_file.open(image_file) as img_data:
                        img_bytes = img_data.read()
                        
                        # PIL로 이미지 열기
                        img = Image.open(io.BytesIO(img_bytes))
                        
                        # 파일명 정리
                        clean_name = Path(image_file).name
                        output_path = output_dir / clean_name
                        
                        # 이미지 저장
                        img.save(output_path, quality=95)
                        
                        # 이미지 정보 수집
                        img_info = {
                            'name': clean_name,
                            'path': str(output_path),
                            'size': img.size,
                            'format': img.format,
                            'mode': img.mode,
                            'file_size': len(img_bytes)
                        }
                        extracted_images.append(img_info)
                        
                        print(f"   ✅ {clean_name} - {img.size[0]}x{img.size[1]} ({img.format})")
                        
                except Exception as e:
                    print(f"   ❌ {image_file} 추출 실패: {e}")
            
            return extracted_images
            
    except Exception as e:
        print(f"❌ 엑셀 파일 열기 실패: {e}")
        return []

def main():
    """메인 함수"""
    print("🚀 엑셀 파일 이미지 추출 시스템")
    print("=" * 50)
    
    # 현재 디렉토리의 엑셀 파일들 찾기
    excel_files = list(Path(".").glob("*.xlsx"))
    
    if not excel_files:
        print("❌ 엑셀 파일을 찾을 수 없습니다.")
        return
    
    print(f"📁 발견된 엑셀 파일: {len(excel_files)}개")
    
    all_extracted_images = []
    
    for excel_file in excel_files:
        print(f"\n📊 {excel_file.name} 분석 중...")
        extracted = extract_images_from_excel(excel_file)
        if extracted:
            all_extracted_images.extend(extracted)
    
    print(f"\n🎉 총 {len(all_extracted_images)}개의 이미지 추출 완료!")
    
    if all_extracted_images:
        print("\n📋 추출된 이미지 목록:")
        for i, img in enumerate(all_extracted_images, 1):
            print(f"{i}. {img['name']}")
            print(f"   📁 경로: {img['path']}")
            print(f"   📏 크기: {img['size'][0]}x{img['size'][1]}")
            print(f"   📊 형식: {img['format']} ({img['mode']})")
            print(f"   💾 파일크기: {img['file_size']:,} bytes")
            print()

if __name__ == "__main__":
    main()
