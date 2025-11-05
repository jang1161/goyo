import 'package:flutter/material.dart';
import 'package:goyo_app/ui/home/home_page.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final idCtrl = TextEditingController();
  final pwCtrl = TextEditingController();
  bool obscure = true;
  bool loading = false;

  @override
  void dispose() {
    idCtrl.dispose();
    pwCtrl.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    if (loading) return;
    setState(() => loading = true);

    final id = idCtrl.text.trim();
    final pw = pwCtrl.text;

    await Future.delayed(const Duration(milliseconds: 200)); // UX용 미세 딜레이

    if (id == '1234' && pw == '1234') {
      if (!mounted) return;
       Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => const HomePage()),
      );
    } else {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('잘못된 ID/PW 입니다. (지금은 1234 / 1234)')),
      );
    }

    if (mounted) setState(() => loading = false);
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 420),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // 로고
                  CircleAvatar(
                    radius: 36,
                    backgroundColor: cs.primary.withOpacity(0.15),
                    child: Icon(Icons.hearing, color: cs.primary, size: 36),
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'Welcome to ANC',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 24, fontWeight: FontWeight.w700, color: cs.onSurface),
                  ),
                  const SizedBox(height: 24),

                  // ID
                  Text('ID', style: TextStyle(color: cs.onSurfaceVariant)),
                  const SizedBox(height: 8),
                  TextField(
                    controller: idCtrl,
                    textInputAction: TextInputAction.next,
                    decoration: const InputDecoration(
                      hintText: 'Enter your email',
                      prefixIcon: Icon(Icons.person_outline),
                    ),
                  ),
                  const SizedBox(height: 14),

                  // PW
                  Text('Password', style: TextStyle(color: cs.onSurfaceVariant)),
                  const SizedBox(height: 8),
                  TextField(
                    controller: pwCtrl,
                    obscureText: obscure,
                    onSubmitted: (_) => _submit(), // 엔터로 로그인
                    decoration: InputDecoration(
                      hintText: 'Enter your password',
                      prefixIcon: const Icon(Icons.lock_outline),
                      suffixIcon: IconButton(
                        onPressed: () => setState(() => obscure = !obscure),
                        icon: Icon(obscure ? Icons.visibility_off : Icons.visibility),
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),

                  // 로그인 버튼
                  ElevatedButton(
                    onPressed: loading ? null : _submit,
                    child: loading
                        ? const SizedBox(
                            height: 22, width: 22, child: CircularProgressIndicator(strokeWidth: 2))
                        : const Text('Sign in'),
                  ),
                  const SizedBox(height: 12),

                  // 도움말
                  Text(
                    'Demo: ID 1234 / PW 1234',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: cs.onSurfaceVariant),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}