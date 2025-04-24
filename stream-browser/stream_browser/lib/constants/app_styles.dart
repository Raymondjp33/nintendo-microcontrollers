import 'package:flutter/material.dart';

class AppTextStyles {
  static TextStyle minecraft({double? fontSize}) {
    return TextStyle(
      fontSize: fontSize,
      color: Colors.white,
      fontFamily: 'Minecraft',
    );
  }

  static TextStyle minecraftTen({double? fontSize}) {
    return TextStyle(
      fontSize: fontSize,
      color: Colors.white,
      fontFamily: 'Minecraft-Ten',
    );
  }
}
