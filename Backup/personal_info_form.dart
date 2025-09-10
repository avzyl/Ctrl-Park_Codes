import 'package:flutter/material.dart';
import '../../../../../utils/constants/sizes.dart';
import '../../../../../utils/constants/text_strings.dart';

class PersonalInfoForm extends StatelessWidget {
  const PersonalInfoForm({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        /// First Name
        TextFormField(
          decoration: const InputDecoration(
            labelText: CPTexts.firstName,
            // prefixIcon: Icon(Iconsax.user),
          ),
        ),
        const SizedBox(height: CPSizes.spaceBtwInputFields),

        /// Middle Name
        TextFormField(
          decoration: const InputDecoration(
            labelText: CPTexts.middleName,
            // prefixIcon: Icon(Iconsax.user),
          ),
        ),
        const SizedBox(height: CPSizes.spaceBtwInputFields),

        /// Last Name
        TextFormField(
          decoration: const InputDecoration(
            labelText: CPTexts.lastName,
            // prefixIcon: Icon(Iconsax.user),
          ),
        ),
        const SizedBox(height: CPSizes.spaceBtwInputFields),

        /// Email
        TextFormField(
          decoration: const InputDecoration(
            labelText: CPTexts.emailLabel,
            // prefixIcon: Icon(Iconsax.user),
          ),
        ),
        const SizedBox(height: CPSizes.spaceBtwInputFields),

        /// Phone
        TextFormField(
          decoration: const InputDecoration(
            labelText: CPTexts.contactNo,
            // prefixIcon: Icon(Iconsax.user),
          ),
        ),
        const SizedBox(height: CPSizes.spaceBtwInputFields),

        /// Address
        TextFormField(
          decoration: const InputDecoration(
            labelText: CPTexts.address,
            // prefixIcon: Icon(Iconsax.user),
          ),
        ),
      ],
    );
  }
}
