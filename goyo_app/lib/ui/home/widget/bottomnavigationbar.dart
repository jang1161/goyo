import 'package:flutter/material.dart';

class AppNavigationBar extends StatelessWidget {
  final int currentIndex;
  final ValueChanged<int> onSelected;

  const AppNavigationBar({
    super.key,
    required this.currentIndex,
    required this.onSelected,
  });

  @override
  Widget build(BuildContext context) {
    return NavigationBar(
      selectedIndex: currentIndex,
      onDestinationSelected: onSelected,
      destinations: const [
        NavigationDestination(icon: Icon(Icons.speaker), label: 'Device Manager'),
        NavigationDestination(icon: Icon(Icons.home_outlined), label: 'Home'),
        NavigationDestination(icon: Icon(Icons.person_outline), label: 'Profile'),
      ],
    );
  }
}