import 'package:flutter/material.dart';
import 'package:goyo_app/ui/home/home_page.dart';
import 'package:goyo_app/ui/login/login_page.dart';
import 'package:goyo_app/ui/profile/profile_page.dart';
import 'theme/app_theme.dart';

void main() => runApp(const GoyoApp());

class GoyoApp extends StatelessWidget {
  const GoyoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'GOYO',
      theme: AppTheme.light(), // ✅ 공통 테마
      darkTheme: AppTheme.dark(), // ✅ 다크 테마(선택)
      themeMode: ThemeMode.light, // 시스템/다크 스위칭 가능
      home: LoginPage(),
      initialRoute: '/login',
      routes: {
        '/login': (_) => const LoginPage(),
        '/home': (_) => const HomePage(), // HomePage(initialIndex: 1) 가능
        '/profile': (_) => const ProfilePage(),
      },
    );
  }
}
