\# Contributing to AI Security \& Surveillance System



Thank you for considering contributing to this project!



\## 🤝 Code of Conduct



\- Be respectful and constructive

\- Welcome newcomers and help them learn

\- Focus on what is best for the community

\- Show empathy towards others



\## 🐛 Reporting Bugs



\*\*Before creating a bug report:\*\*

\- Check existing issues to avoid duplicates

\- Collect relevant information (OS, Docker version, logs)



\*\*Bug report template:\*\*

```

Title: Brief description



Description: Detailed explanation



Steps to Reproduce:

1\. Step one

2\. Step two

3\. Expected vs actual result



Environment:

\- OS: Windows/Mac/Linux

\- Docker: version

\- Python: version (if local)



Error logs:

\[Paste error here]

```



\## 💡 Suggesting Enhancements



\*\*Enhancement template:\*\*

```

Title: Feature name



Use Case: Why is this needed?



Proposed Solution: How should it work?



Alternatives: Other options considered

```



\## 🔧 Development Setup



\### Prerequisites

\- Python 3.11+

\- Docker Desktop

\- Git



\### Setup

```bash

\# Clone your fork

git clone https://github.com/YOUR\_USERNAME/ai-security-surveillance-system.git

cd ai-security-surveillance-system



\# Install backend dependencies

cd backend

pip install -r requirements.txt

python download\_models.py



\# Install frontend dependencies

cd ../frontend

pip install -r requirements.txt

```



\### Running Locally

```bash

\# Terminal 1: Backend

cd backend

uvicorn app:app --reload



\# Terminal 2: Frontend

cd frontend

streamlit run dashboard\_simple.py

```



\## 📝 Code Style



\### Python

\- Follow \*\*PEP 8\*\*

\- Use \*\*type hints\*\*

\- Write \*\*docstrings\*\*

\- Keep functions \*\*small\*\* (<50 lines)

\- Use \*\*meaningful names\*\*



\*\*Example:\*\*

```python

def detect\_faces(frame: np.ndarray, confidence: float = 0.5) -> list\[dict]:

&nbsp;   """

&nbsp;   Detect faces in video frame.

&nbsp;   

&nbsp;   Args:

&nbsp;       frame: Input image (BGR format)

&nbsp;       confidence: Minimum confidence (0-1)

&nbsp;       

&nbsp;   Returns:

&nbsp;       List of detected faces with boxes and scores

&nbsp;   """

&nbsp;   pass

```



\### Commit Messages



Follow \*\*Conventional Commits\*\*:



\- `feat:` - New feature

\- `fix:` - Bug fix

\- `docs:` - Documentation

\- `style:` - Formatting

\- `refactor:` - Code refactor

\- `test:` - Tests

\- `chore:` - Maintenance



\*\*Examples:\*\*

```

feat(face-recognition): add liveness detection

fix(api): resolve JWT expiration bug

docs(readme): update installation steps

```



\## 🧪 Testing

```bash

\# Run all tests

pytest



\# With coverage

pytest --cov=backend

```



\## 📚 Documentation



\- Add \*\*docstrings\*\* to all public functions

\- Update \*\*README.md\*\* for new features

\- Update \*\*API docs\*\* if changing endpoints

\- Add \*\*inline comments\*\* for complex logic



\## 🚀 Pull Request Process



1\. \*\*Fork\*\* the repository

2\. \*\*Create\*\* feature branch (`git checkout -b feature/AmazingFeature`)

3\. \*\*Commit\*\* changes (`git commit -m 'feat: add AmazingFeature'`)

4\. \*\*Push\*\* to branch (`git push origin feature/AmazingFeature`)

5\. \*\*Open\*\* Pull Request



\### PR Checklist

\- \[ ] Code follows style guidelines

\- \[ ] Tests added and passing

\- \[ ] Documentation updated

\- \[ ] No new warnings

\- \[ ] Self-review completed



\## 💬 Questions?



\- Check existing issues first

\- Open a new issue for questions

\- Email: daneaudreyy24@gmail.com



\## 🏆 Recognition



Contributors will be recognized in:

\- README acknowledgments

\- Release notes

\- Project documentation



\## 📜 License



By contributing, you agree that your contributions will be licensed under the MIT License.



---



Thank you for contributing! 🎉

