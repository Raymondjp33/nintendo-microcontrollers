import 'package:flutter/material.dart';

import '../constants/app_styles.dart';
import '../widgets/spacing.dart';

class MiddleContent extends StatelessWidget {
  const MiddleContent({super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(
          '12:25PM EST',
          style: AppTextStyles.minecraftTen(fontSize: 32),
        ),
        VerticalSpace(20),
        Text(
          'Stream Runtime',
          style: AppTextStyles.minecraft(fontSize: 32),
        ),
        Text(
          '700d 22h',
          style: AppTextStyles.minecraft(fontSize: 32),
        ),
        VerticalSpace(10),
        Text(
          'IM AWAY',
          style: AppTextStyles.minecraftTen(fontSize: 64),
        ),
        Text(
          'Checkout what commands are available with !help',
          textAlign: TextAlign.center,
          style: AppTextStyles.minecraft(fontSize: 24),
        ),
      ],
    );
  }
}
