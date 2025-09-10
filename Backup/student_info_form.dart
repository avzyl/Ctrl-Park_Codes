import 'package:flutter/material.dart';
import '../../../../../utils/constants/sizes.dart';
import '../../../../../utils/constants/text_strings.dart';

class StudentInfoForm extends StatelessWidget {
  const StudentInfoForm({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        /// Full Name
        TextFormField(
          decoration: const InputDecoration(
            labelText: CPTexts.fullName,
            // prefixIcon: Icon(Iconsax.direct),
          ),
        ),
        const SizedBox(height: CPSizes.spaceBtwInputFields),

        /// Student Number
        TextFormField(
          decoration: const InputDecoration(
            labelText: CPTexts.studentNo,
            // prefixIcon: Icon(Iconsax.call),
          ),
        ),
        const SizedBox(height: CPSizes.spaceBtwInputFields),

        /// Department
        TextFormField(
          decoration: const InputDecoration(
            labelText: CPTexts.dept,
            // prefixIcon: Icon(Iconsax.card),
          ),
        ),
        const SizedBox(height: CPSizes.spaceBtwInputFields),

        /// Program
        TextFormField(
          decoration: const InputDecoration(
            labelText: CPTexts.program,
            // prefixIcon: Icon(Iconsax.card),
          ),
        ),
        const SizedBox(height: CPSizes.spaceBtwInputFields),

        /// Year
        TextFormField(
          decoration: const InputDecoration(
            labelText: CPTexts.yearLvl,
            // prefixIcon: Icon(Iconsax.card),
          ),
        ),
        const SizedBox(height: CPSizes.spaceBtwInputFields),

        /// Section
        TextFormField(
          decoration: const InputDecoration(
            labelText: CPTexts.section,
            // prefixIcon: Icon(Iconsax.card),
          ),
        ),
        const SizedBox(height: CPSizes.spaceBtwInputFields),
      ],
    );
  }
}
