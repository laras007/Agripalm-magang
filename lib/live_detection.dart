import 'package:ultralytics_yolo/yolo_view.dart';
import 'dart:io';
//import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'dart:convert';
import 'package:path_provider/path_provider.dart';
import 'package:geolocator/geolocator.dart';
import 'package:flutter/foundation.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_result.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

class CameraDetectionScreen extends StatefulWidget {
  const CameraDetectionScreen({super.key});

  @override
  State<CameraDetectionScreen> createState() => _CameraDetectionScreenState();
}

class _CameraDetectionScreenState extends State<CameraDetectionScreen> {
  late YOLOViewController controller;
  final GlobalKey _repaintKey = GlobalKey();
  late YOLOStreamingConfig _streamingConfig;
  List<dynamic> currentResults = [];
  double fps = 0.0;
  double processingTimeMs = 0.0;
  Uint8List? _lastFrameBytes;
  Map<int, int> classCounts = {};
  final List<String> classLabels = [
    'Janjang kosong', // 0
    'Kurang masak', // 1
    'TBS abnormal', // 2
    'TBS masak', // 3
    'TBS mentah', // 4
    'Terlalu masak', // 5
  ];
  // (removed unused _lastResultTimestampMs)
  // timestamps (ms) of recent onResult callbacks for local FPS calculation
  final List<int> _frameTimestampsMs = [];
  // smoothing factor for displayed FPS (0..1)
  final double _fpsSmoothing = 0.2;

  @override
  void initState() {
    super.initState();
    controller = YOLOViewController();
    _streamingConfig = YOLOStreamingConfig(
      includeDetections: true,
      includeFps: true,
      includeProcessingTimeMs: true,
      includeOriginalImage: true,
    );
  }

  // Fungsi untuk mengirim data ke API dengan format JSON khusus
  Future<void> sendDataToApi(Position? pos, Map<int, int> counts) async {
    // ⚠️ GANTI URL INI dengan alamat API server Anda yang asli
    final url = Uri.parse('http://10.160.157.180:3000/api/sawit'); 

    try {
      // Tampilkan pesan loading
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Mengirim data ke server...')),
      );

      // Susun data sesuai format yang diminta
      Map<String, dynamic> dataKirim = {
        "latitude": pos?.latitude ?? 0.0,
        "longitude": pos?.longitude ?? 0.0,
        // Mapping manual Index ke Nama Key
        "janjang_kosong": counts[0] ?? 0,
        "kurang_masak": counts[1] ?? 0,
        "tbs_abnormal": counts[2] ?? 0,
        "tbs_masak": counts[3] ?? 0,
        "tbs_mentah": counts[4] ?? 0,
        "terlalu_masak": counts[5] ?? 0,
      };

      // Kirim Request POST
      final response = await http.post(
        url,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: jsonEncode(dataKirim),
      );

      // Cek Hasil
      if (response.statusCode == 200 || response.statusCode == 201) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(backgroundColor: Colors.green, content: Text('✅ Data Terkirim!')),
        );
        print('Sukses: ${response.body}');
      } else {
        throw Exception('Gagal: ${response.statusCode}');
      }
    } catch (e) {
      print('Error kirim API: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(backgroundColor: Colors.red, content: Text('Gagal kirim: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Camera view with YOLO processing
          RepaintBoundary(
            key: _repaintKey,
            child: YOLOView(
              modelPath: 'yolo11n',
              task: YOLOTask.detect,
              streamingConfig: _streamingConfig,
              showNativeUI: true,
              controller: controller,
              onResult: (dynamic results) {
                if (kDebugMode) {
                  try {
                    print(
                      'onResult called - runtimeType=${results.runtimeType}',
                    );
                    if (results is List) {
                      print('onResult list length=${results.length}');
                      if (results.isNotEmpty)
                        print('onResult[0]=${results[0]}');
                    } else if (results is Map) {
                      print('onResult map keys=${results.keys}');
                    } else {
                      print('onResult value=$results');
                    }
                  } catch (e) {
                    print('onResult debug print failed: $e');
                  }
                }
                // compute local FPS using sliding window of onResult timestamps
                final int nowMs = DateTime.now().millisecondsSinceEpoch;
                _frameTimestampsMs.add(nowMs);
                // remove timestamps older than 1 second
                _frameTimestampsMs.removeWhere((t) => t < nowMs - 1000);
                final double instantFps = _frameTimestampsMs.length.toDouble();

                // normalize results to a List<dynamic> so we can count them reliably
                List<dynamic> parsed = [];
                try {
                  if (results is List) {
                    parsed = List<dynamic>.from(results);
                  } else if (results is Map &&
                      results.containsKey('boxes') &&
                      results['boxes'] is List) {
                    parsed = List<dynamic>.from(results['boxes']);
                  } else if (results is Map &&
                      results.containsKey('detections') &&
                      results['detections'] is List) {
                    parsed = List<dynamic>.from(results['detections']);
                  } else if (results is Iterable) {
                    parsed = List<dynamic>.from(results);
                  } else {
                    // single result -> wrap
                    parsed = [results];
                  }
                } catch (e) {
                  parsed = [results];
                }

                if (kDebugMode) {
                  try {
                    print('parsed results length=${parsed.length}');
                    for (var i = 0; i < parsed.length && i < 3; i++) {
                      final e = parsed[i];
                      if (e is YOLOResult) {
                        print(
                          'parsed[$i] class=${e.className} conf=${e.confidence}',
                        );
                      } else if (e is Map) {
                        print('parsed[$i] map keys=${e.keys}');
                      } else {
                        print('parsed[$i] value=$e');
                      }
                    }
                  } catch (e) {
                    print('parsed debug failed: $e');
                  }
                }

                if (!mounted) return;
                setState(() {
                  currentResults = parsed;
                  // compute counts per class index
                  final Map<int, int> counts = {};
                  for (var item in parsed) {
                    int? idx;
                    if (item is YOLOResult) {
                      idx = item.classIndex;
                    } else if (item is Map && item.containsKey('classIndex')) {
                      try {
                        final v = item['classIndex'];
                        if (v is int)
                          idx = v;
                        else if (v is num)
                          idx = v.toInt();
                      } catch (_) {
                        idx = null;
                      }
                    }

                    if (idx != null) {
                      counts[idx] = (counts[idx] ?? 0) + 1;
                    }
                  }
                  classCounts = counts;
                  // apply simple exponential smoothing so FPS display is stable
                  if (instantFps > 0) {
                    fps =
                        (fps * (1 - _fpsSmoothing)) +
                        (instantFps * _fpsSmoothing);
                  }
                });
              },
              onPerformanceMetrics: (metrics) {
                // store processing time and only accept plugin FPS when > 0
                if (!mounted) return;
                setState(() {
                  processingTimeMs = metrics.processingTimeMs;
                  if (metrics.fps > 0) {
                    fps = metrics.fps;
                  }
                });
                // keep console logs for debugging (use debugPrint to satisfy lints)
                if (kDebugMode) {
                  print('FPS: ${metrics.fps.toStringAsFixed(1)}');
                  print(
                    'Processing time: ${metrics.processingTimeMs.toStringAsFixed(1)}ms',
                  );
                }
              },
              onStreamingData: (stream) async {
                // stream contains detections, fps, processingTimeMs, originalImage etc.
                try {
                  final Map? map = stream as Map?;
                  if (map != null) {
                    if (map.containsKey('originalImage') &&
                        map['originalImage'] != null) {
                      final img = map['originalImage'];
                      if (img is Uint8List) {
                        _lastFrameBytes = img;
                      }
                    }

                    // Also handle detections to keep currentResults in sync
                    if (map.containsKey('detections') &&
                        map['detections'] is List) {
                      final detections = List<dynamic>.from(map['detections']);
                      if (!mounted) return;
                      setState(() {
                        currentResults = detections;
                        // compute classCounts
                        final Map<int, int> counts = {};
                        for (var d in detections) {
                          int? idx;
                          if (d is Map && d.containsKey('classIndex')) {
                            final v = d['classIndex'];
                            if (v is int)
                              idx = v;
                            else if (v is num)
                              idx = v.toInt();
                          }
                          if (idx != null) counts[idx] = (counts[idx] ?? 0) + 1;
                        }
                        classCounts = counts;
                      });
                    }
                  }
                } catch (e) {
                  print('onStreamingData error: $e');
                }
              },
            ),
          ),

          // Overlay UI
          Positioned(
            top: 50,
            left: 20,
            child: Container(
              padding: EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    'Objects: ${currentResults.length}',
                    style: TextStyle(color: Colors.white, fontSize: 18),
                  ),
                  SizedBox(height: 4),
                  Text(
                    'FPS: ${fps.toStringAsFixed(1)}',
                    style: TextStyle(color: Colors.white70, fontSize: 14),
                  ),
                  SizedBox(height: 8),
                  // class counts
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: List.generate(classLabels.length, (i) {
                      final count = classCounts[i] ?? 0;
                      return Text(
                        '$i: ${classLabels[i]} — $count',
                        style: TextStyle(color: Colors.white70, fontSize: 12),
                      );
                    }),
                  ),
                ],
              ),
            ),
          ),
          // Capture button bottom center
          Positioned(
            bottom: 30,
            left: 0,
            right: 0,
            child: Center(
              child: FloatingActionButton(
                backgroundColor: Colors.blue, // Biar terlihat jelas
                child: const Icon(Icons.send), // Ganti ikon biar sesuai konteks kirim
                onPressed: () async {
                  // 1. Cek apakah ada frame/deteksi (opsional, untuk safety)
                  if (currentResults.isEmpty) {
                    ScaffoldMessenger.of(context).showSnackBar(
                       const SnackBar(content: Text('Tidak ada objek terdeteksi untuk dikirim.')),
                    );
                    // Jika ingin tetap kirim walau kosong, hapus return ini
                    return; 
                  }

                  // 2. Ambil Lokasi GPS
                  Position? pos;
                  try {
                    bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
                    if (!serviceEnabled) {
                      throw Exception('Layanan Lokasi Mati');
                    }

                    LocationPermission permission = await Geolocator.checkPermission();
                    if (permission == LocationPermission.denied) {
                      permission = await Geolocator.requestPermission();
                    }
                    
                    if (permission == LocationPermission.denied || 
                        permission == LocationPermission.deniedForever) {
                      throw Exception('Izin Lokasi Ditolak');
                    }

                    // Ambil posisi saat ini
                    pos = await Geolocator.getCurrentPosition(
                      desiredAccuracy: LocationAccuracy.high // Akurasi tinggi
                    );
                  } catch (e) {
                    print('Gagal ambil lokasi: $e');
                    // Tetap lanjut kirim data walau lokasi null (akan jadi 0.0)
                  }

                  // 3. LANGSUNG KIRIM KE API (Panggil fungsi tadi)
                  await sendDataToApi(pos, classCounts);
                  
                  // Simpan data ke CSV lokal
                  try {
                    String filename = 'Palm_${DateTime.now().millisecondsSinceEpoch}.csv';
                    File file;
                    if (Platform.isAndroid) {
                      Directory? extDir = await getExternalStorageDirectory();
                      if (extDir != null) {
                        final appDir = Directory('${extDir.path}/DeteksiSawit');
                        if (!await appDir.exists()) {
                          await appDir.create(recursive: true);
                        }
                        file = File('${appDir.path}/$filename');
                      } else {
                        // fallback ke dokumen
                        final directory = await getApplicationDocumentsDirectory();
                        file = File('${directory.path}/$filename');
                      }
                    } else {
                      final directory = await getApplicationDocumentsDirectory();
                      file = File('${directory.path}/$filename');
                    }

                    // Susun data sesuai format CSV
                    String csvRow =
                        '${DateTime.now().toIso8601String()},${pos?.latitude ?? 0.0},${pos?.longitude ?? 0.0},${classCounts[0] ?? 0},${classCounts[1] ?? 0},${classCounts[2] ?? 0},${classCounts[3] ?? 0},${classCounts[4] ?? 0},${classCounts[5] ?? 0}\n';

                    // Tambahkan header dan data
                    await file.writeAsString(
                      'waktu,latitude,longitude,janjang_kosong,kurang_masak,tbs_abnormal,tbs_masak,tbs_mentah,terlalu_masak\n',
                      mode: FileMode.write,
                    );
                    await file.writeAsString(csvRow, mode: FileMode.append);
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('Data disimpan ke history!')),
                    );
                  } catch (e) {
                    print('Gagal simpan CSV: $e');
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text('Gagal simpan CSV: $e')),
                    );
                  }
                },
              ),
            ),
          ),
        ],
      ),
    );
  }
}