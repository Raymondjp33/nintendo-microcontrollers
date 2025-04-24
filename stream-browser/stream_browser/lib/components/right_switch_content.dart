import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../constants/app_styles.dart';
import '../services/file_provider.dart';
import '../widgets/spacing.dart';
import 'line_item.dart';

class RightSwitchContent extends StatelessWidget {
  const RightSwitchContent({super.key});

  @override
  Widget build(BuildContext context) {
    int? enc = context.watch<FileProvider>().switch1Data?['pokemon_encounters']
        ?[0]?['encounters'];
    print('rerendering $enc');
    return Column(
      crossAxisAlignment: CrossAxisAlignment.end,
      children: [
        Text(
          'POKEMON SWORD',
          style: AppTextStyles.minecraftTen(fontSize: 32),
        ),
        LineItem(leftText: 'Odds', rightText: '1/100'),
        LineItem(leftText: 'Normal shinies', rightText: '60'),
        LineItem(leftText: 'Legendary shinies', rightText: '15'),
        LineItem(leftText: 'Total dens', rightText: '2500'),
        LineItem(leftText: 'Average checks', rightText: '92.25'),
        Spacer(),
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 150,
              height: 108,
              child: Image.asset(
                'assets/images/637.gif',
                fit: BoxFit.contain,
              ),
            ),
            HorizontalSpace(10),
            Flexible(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisAlignment: MainAxisAlignment.start,
                children: [
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        '$enc',
                        style: AppTextStyles.minecraft(fontSize: 48),
                      ),
                      Spacer(),
                      Column(
                        mainAxisSize: MainAxisSize.min,
                        crossAxisAlignment: CrossAxisAlignment.end,
                        children: [
                          Text(
                            'Total dens',
                            style: AppTextStyles.minecraft(fontSize: 24),
                          ),
                          Text(
                            '300',
                            style: AppTextStyles.minecraft(fontSize: 24),
                          ),
                        ],
                      ),
                    ],
                  ),
                  Text(
                    '1d 2h 10m 5s',
                    style: AppTextStyles.minecraft(fontSize: 24),
                  ),
                ],
              ),
            ),
          ],
        ),
      ],
    );
  }
}
