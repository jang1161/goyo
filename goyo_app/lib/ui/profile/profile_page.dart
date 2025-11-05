import 'package:flutter/material.dart';

/// Profile: 사용자 이름 + 선호 모드(Study/Sleep/Focus) + 모드별 ANC 프리셋(UI 더미)
class ProfilePage extends StatefulWidget {
  const ProfilePage({super.key});

  @override
  State<ProfilePage> createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  final nameCtrl = TextEditingController(text: 'Lee Wongyu');

  // 현재 선택된 모드
  String currentMode = 'focus'; // 'study' | 'sleep' | 'focus'

  // 모드별 ANC 프리셋(더미 데이터): intensity(0~1), auto(true/false)
  final Map<String, AncPreset> presets = {
    'study': AncPreset(intensity: 0.6, auto: true),
    'sleep': AncPreset(intensity: 0.4, auto: true),
    'focus': AncPreset(intensity: 0.8, auto: false),
  };

  bool saving = false;

  @override
  void dispose() {
    nameCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final preset = presets[currentMode]!;

    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            // ── 사용자 카드 ───────────────────────────────────────────────
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: [
                    CircleAvatar(
                      radius: 28,
                      backgroundColor: cs.primary.withOpacity(.15),
                      child: Icon(Icons.person, color: cs.primary, size: 28),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: TextField(
                        controller: nameCtrl,
                        decoration: const InputDecoration(
                          labelText: 'Your name',
                          hintText: 'Enter your display name',
                          prefixIcon: Icon(Icons.badge_outlined),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),

            // ── 선호 모드 선택 (SegmentedButton) ────────────────────────
            Text(
              'Preferred sound mode',
              style: TextStyle(
                fontWeight: FontWeight.w700,
                color: cs.onSurface,
              ),
            ),
            const SizedBox(height: 8),
            SegmentedButton<String>(
              segments: const [
                ButtonSegment(
                  value: 'study',
                  label: Text('Study'),
                  icon: Icon(Icons.menu_book_outlined),
                ),
                ButtonSegment(
                  value: 'sleep',
                  label: Text('Sleep'),
                  icon: Icon(Icons.nightlight_outlined),
                ),
                ButtonSegment(
                  value: 'focus',
                  label: Text('Focus'),
                  icon: Icon(Icons.center_focus_weak),
                ),
              ],
              selected: {currentMode},
              onSelectionChanged: (s) => setState(() => currentMode = s.first),
            ),
            const SizedBox(height: 16),

            // ── 현재 모드의 ANC 프리셋 ──────────────────────────────────
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.hearing, color: cs.primary),
                        const SizedBox(width: 8),
                        Text(
                          'ANC preset for "${currentMode.toUpperCase()}"',
                          style: TextStyle(
                            fontWeight: FontWeight.w700,
                            color: cs.onSurface,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    SwitchListTile(
                      contentPadding: EdgeInsets.zero,
                      value: preset.auto,
                      onChanged: (v) => setState(
                        () => presets[currentMode] = preset.copyWith(auto: v),
                      ),
                      title: const Text('Automatic mode'),
                      subtitle: const Text(
                        'Adjust suppression based on ambient noise',
                      ),
                    ),
                    const SizedBox(height: 8),
                    ListTile(
                      contentPadding: EdgeInsets.zero,
                      title: const Text('Suppression intensity'),
                      subtitle: Slider(
                        value: preset.intensity,
                        onChanged: (v) => setState(
                          () => presets[currentMode] = preset.copyWith(
                            intensity: v,
                          ),
                        ),
                        min: 0.0,
                        max: 1.0,
                      ),
                      trailing: Text('${(preset.intensity * 100).round()}%'),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),

            // ── 데이터/로그(더미) ────────────────────────────────────────
            Card(
              child: ListTile(
                leading: const Icon(Icons.bar_chart_outlined),
                title: const Text('Usage & noise stats'),
                subtitle: const Text(
                  'View noise patterns and suppression history (demo)',
                ),
                trailing: const Icon(Icons.chevron_right),
                onTap: () {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text('Demo: stats page is not implemented yet.'),
                    ),
                  );
                },
              ),
            ),
            const SizedBox(height: 20),

            // ── 저장 버튼(더미) ─────────────────────────────────────────
            FilledButton(
              onPressed: saving ? null : _save,
              child: saving
                  ? const SizedBox(
                      height: 22,
                      width: 22,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text('Save changes'),
            ),
            const SizedBox(height: 10),
            FilledButton.tonalIcon(
              onPressed: () {
                Navigator.of(
                  context,
                ).pushNamedAndRemoveUntil('/login', (route) => false);
              },

              label: const Text('Log out'),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _save() async {
    setState(() => saving = true);
    await Future.delayed(const Duration(milliseconds: 300)); // 데모용
    if (!mounted) return;
    setState(() => saving = false);
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Saved (demo: local state only)')),
    );
  }
}

class AncPreset {
  final double intensity; // 0..1
  final bool auto;
  const AncPreset({required this.intensity, required this.auto});

  AncPreset copyWith({double? intensity, bool? auto}) => AncPreset(
    intensity: intensity ?? this.intensity,
    auto: auto ?? this.auto,
  );
}
