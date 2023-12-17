import { PipeTransform, Injectable, ArgumentMetadata, HttpException, HttpStatus } from "@nestjs/common";

@Injectable()
export class PdfFileMimeTypeValidationPipe implements PipeTransform {
  transform(values: any[], metadata: ArgumentMetadata) {
    if (values) {
      let errors: any[] = [];
      for (let i = 0; i < values.length; i++) {
        const file = values[i];

        file.originalname = Buffer.from(file.originalname, 'latin1').toString(
          'utf8',
        );

        if (
          file.mimetype !== "application/pdf") {
          errors.push({ text: "Unauthorized file format loaded", file: file });
        }

        if ((file.size / 1000000) > 200) {
          errors.push({ text: "File size too large > 200mb", file: file });
        }
      }

      if (errors.length === 0) {
        return values;
      } else {
        throw new HttpException({
          status: HttpStatus.UNSUPPORTED_MEDIA_TYPE,
          error: errors.map(er => ({
            text: er.text, file: {
              originalname: er.file.originalname,
              mimetype: er.file.mimetype,
              size: er.file.size
            }
          }))
        }, HttpStatus.UNSUPPORTED_MEDIA_TYPE);
      }
    }
  }
}