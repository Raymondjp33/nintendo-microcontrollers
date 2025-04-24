import 'package:flutter/material.dart';

import '../constants/app_styles.dart';

class LineItem extends StatelessWidget {
  const LineItem({
    required this.leftText,
    required this.rightText,
    super.key,
  });

  final String leftText;
  final String rightText;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Text(
          leftText,
          style: AppTextStyles.minecraft(fontSize: 24),
        ),
        Spacer(),
        Text(
          rightText,
          style: AppTextStyles.minecraft(fontSize: 24),
        ),
      ],
    );
  }
}
