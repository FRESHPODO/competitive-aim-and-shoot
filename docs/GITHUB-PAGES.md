# GitHub Pages 실행 가능성 검토

검토일: 2026-10-20. 현재 Windows 구현 및 Unity 6000.3 공식 문서 기준이며 Web 빌드 실행을 검증한 결과는 아닙니다.

## 결론

**현재 파일을 그대로 Pages에 올려 실시간 AI 시뮬레이터를 실행할 수는 없습니다.** Unity 화면은 Web 빌드로 배포할 수 있지만, Python/PyTorch 추론과 공유 표적/피해 계산을 실행하는 서버가 따로 필요합니다. Pages는 정적 HTML/CSS/JavaScript 호스팅이며 Python 서버를 실행하지 않습니다.

## 현재 구조에서 걸리는 부분

| 부분 | 현재 구현 | 웹 전환 시 필요 작업 |
|---|---|---|
| 표시 | Windows Unity 실행 파일 | Unity Web/WebGL 빌드 타깃과 Web Build Support 설치 (현재 PC에는 Windows 모듈만 설치됨) |
| AI와 과제 | Python, NumPy/SciPy, PyTorch, Stable-Baselines3 | 외부 HTTPS 추론 서버 또는 전체 브라우저 이식 |
| 연결 | `http://127.0.0.1:8765` 상수 | HTTPS API 주소 설정, CORS, OPTIONS 처리 |
| 세션 | 서버 내 공용 DualSession 하나 | 방문자별 세션 분리, 동시 요청 제어, 서버 자원 제한 |
| 폰트 | Windows Malgun Gothic 시스템 폰트 | 한글 글꼴 자산 포함 및 웹 표시 검증 |
| CSV | 로컬 파일 AppendAllText | 브라우저 다운로드 또는 사용자별 서버 저장 |
| 배포 압축 | Windows 전용 빌더 | Web 압축 해제 fallback 또는 비압축 설정, MIME/로딩 검증 |

다른 사용자의 브라우저에서 127.0.0.1은 개발자 PC가 아니라 **방문자 자신의 PC**입니다. 현재 서버는 로컬 전용이며 CORS·외부 접속·사용자별 세션을 제공하지 않습니다. 단순히 주소와 바인딩만 공개 서버로 바꾸는 것으로는 충분하지 않습니다.

## 가능한 구성

1. **Pages + 외부 Python HTTPS 서버**: Unity Web 화면을 Pages에서 제공하고 현재 추론/과제 코드를 외부 서버에서 실행합니다. 기존 모델을 유지하기 가장 직접적인 경로입니다. 세션 분리와 지연 측정이 필요합니다. 두 모델의 공통 시각·추론 장벽은 계속 서버에서 유지하고, 네트워크 지연과 시뮬레이션 시간을 구분해야 합니다.
2. **Pages만으로 자급 실행**: 정책을 ONNX/WebAssembly 등 브라우저 추론 형식으로 변환하고, 지각·운동 계획·시선·이동·피해 환경 전체를 브라우저 또는 Unity C#으로 이식해야 합니다. SAC의 사용 중인 사용자 정의 정책 변환 가능성과 원본 대비 수치·행동 일치성을 먼저 검증해야 합니다. 모델 파일만 변환하면 완료되는 작업이 아닙니다.
3. **기록 재생 페이지**: 사전 계산한 궤적·발사 사건을 Pages에서 재생할 수 있습니다. 별도 서버가 필요 없지만 실시간 파라미터 변경에 따른 새로운 판단은 할 수 없습니다.

현재 목적이 실시간 조건 변경 관찰이면 1번이 구현 부담이 가장 작습니다. 서버 없이 배포하는 것이 필수이면 2번은 별도 이식 작업으로 다루는 편이 적절합니다. 이번 업로드에서는 Web 배포나 외부 서버 공개를 수행하지 않았습니다.

## 공식 근거

- [GitHub Pages 소개](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages): 정적 사이트 호스팅.
- [Pages 사이트 생성](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site): Python 등 서버 언어 미지원.
- [Unity 6000.3 Web networking](https://docs.unity3d.com/6000.3/Documentation/Manual/webgl-networking.html): UnityWebRequest가 브라우저 Fetch를 사용하며 교차 출처 API에는 CORS가 필요함.
- [Unity 6000.3 Web 배포](https://docs.unity3d.com/6000.3/Documentation/Manual/webgl-deploying.html): 압축과 응답 헤더, decompression fallback 설정.
