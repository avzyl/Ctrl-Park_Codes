import 'package:flutter/material.dart';

void main() {
  runApp(const ParkingApp());
}

class ParkingApp extends StatelessWidget {
  const ParkingApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: ParkingLot(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class ParkingLot extends StatefulWidget {
  const ParkingLot({super.key});

  @override
  State<ParkingLot> createState() => _ParkingLotState();
}

class _ParkingLotState extends State<ParkingLot> {
  final int columns = 3;
  final int slotsPerColumn = 15;

  late List<bool> occupied;

  @override
  void initState() {
    super.initState();
    occupied = List.generate(columns * slotsPerColumn, (_) => false);
  }

  void _toggleSlot(int index) {
    setState(() {
      occupied[index] = !occupied[index];
    });
  }

  void _showCarInfo(int index, String label) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text("Car Info"),
        content: Text("Slot $label is occupied by: Toyota ABC-123"),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text("Close"),
          ),
        ],
      ),
    );
  }

  Widget _buildSlot(int index, String label) {
    return GestureDetector(
      onTap: () {
        if (occupied[index]) {
          _showCarInfo(index, label);
        } else {
          _toggleSlot(index);
        }
      },
      child: Container(
        margin: const EdgeInsets.all(6),
        width: 150,
        height: 70,
        decoration: BoxDecoration(
          color: occupied[index] ? Colors.red[800] : Colors.green,
          borderRadius: BorderRadius.circular(8),
        ),
        child: Center(
          child: occupied[index]
              ? const Icon(Icons.directions_car, color: Colors.white, size: 32)
              : Text(
            label,
            style: const TextStyle(color: Colors.white, fontSize: 14),
            textAlign: TextAlign.center,
          ),
        ),
      ),
    );
  }

  Widget _buildDriveway({bool showText = false}) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 8),
      width: 65,
      color: Colors.transparent,
      child: Center(
        child: showText
            ? const Text("Driveway", style: TextStyle(color: Colors.black))
            : null,
      ),
    );
  }

  // Helper to get the prefix per column index
  String _getPrefix(int colIndex) {
    switch (colIndex) {
      case 0:
        return 'S';
      case 1:
        return 'EB';
      case 2:
        return 'EA';
      default:
        return 'C${colIndex + 1}'; // fallback for more columns
    }
  }

  Widget _buildColumn(int colIndex) {
    final prefix = _getPrefix(colIndex);
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: List.generate(
        slotsPerColumn,
            (i) => _buildSlot(colIndex * slotsPerColumn + i, "$prefix${(i + 1).toString().padLeft(2, '0')}"),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Parking Lot Mockup")),
      body: SingleChildScrollView(
        scrollDirection: Axis.vertical,
        child: SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildColumn(0),
              _buildDriveway(),
              _buildColumn(1),
              _buildDriveway(showText: true),
              _buildColumn(2),
            ],
          ),
        ),
      ),
    );
  }
}
