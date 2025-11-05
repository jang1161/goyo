import 'package:flutter/material.dart';
import 'package:goyo_app/ui/device/devicemanager_page.dart';
import 'package:goyo_app/ui/home/widget/bottomnavigationbar.dart';
import 'package:goyo_app/ui/home/widget/hometab.dart';
import 'package:goyo_app/ui/profile/profile_page.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key, this.initialIndex = 1});
  final int initialIndex;
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  late int _idx = widget.initialIndex;

  late final List<Widget> _tabs = const [
    DeviceManager(),
    HomeTab(), // ✅ ANC 토글 + 소음 규칙 리스트
    ProfilePage(), // ✅ 프로필
  ];

  static const _titles = ['Device Manager', 'Home', 'Profile'];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(_titles[_idx])),
      body: IndexedStack(index: _idx, children: _tabs), // 탭 상태 유지
      bottomNavigationBar: AppNavigationBar(
        currentIndex: _idx,
        onSelected: (i) => setState(() => _idx = i),
      ),
    );
  }
}
