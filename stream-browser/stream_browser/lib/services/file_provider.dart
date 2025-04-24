import 'package:flutter/material.dart';
import 'package:socket_io_client/socket_io_client.dart' as io;

class FileProvider with ChangeNotifier {
  FileProvider() {
    connectSocket();
  }

  io.Socket? socket;

  Map<String, dynamic>? switch1Data;
  Map<String, dynamic>? switch2Data;
  Map<String, dynamic>? streamData;

  void connectSocket() {
    socket = io.io(
      'http://localhost:5050',
      io.OptionBuilder()
          .setTransports(['websocket'])
          .disableAutoConnect()
          .build(),
    );

    socket!.onConnect((_) {
      print('[Connected] to server');
    });

    socket!.on('switch1_data', (data) {
      try {
        switch1Data = data as Map<String, dynamic>;
        notifyListeners();
      } catch (e) {
        print('Error with data $e');
      }
    });

    socket!.on('switch2_data', (data) {
      try {
        switch2Data = data as Map<String, dynamic>;
        notifyListeners();
      } catch (e) {
        print('Error with data $e');
      }
    });

    socket!.on('stream_data', (data) {
      try {
        streamData = data as Map<String, dynamic>;
        notifyListeners();
      } catch (e) {
        print('Error with data $e');
      }
    });

    socket!.onDisconnect((_) => print('[Disconnected]'));

    socket!.connect();
  }
}
